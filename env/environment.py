import random
from .models import Action, Observation
from .tasks import TASKS, TASK_INDEX
from .reward_config import REWARD_CONFIG
from .grader import calculate_reward, grade_episode


class CustomerSupportEnv:
    """
    OpenEnv — AI Customer Support Training Environment.

    Follows the standard RL loop:
        obs            = env.reset()
        obs, r, done, info = env.step(action)
    """

    # ---------------------------------------------------------------- #
    #  Construction                                                      #
    # ---------------------------------------------------------------- #
    def __init__(self, task_id: str = None, random_task: bool = True):
        """
        Parameters
        ----------
        task_id     : Pin to a specific task. If None, random_task decides.
        random_task : Pick a random task on each reset() call.
        """
        self.task_id     = task_id
        self.random_task = random_task

        # All internal state is initialised here so the type checker
        # can see it. Actual values are set in reset().
        self._task:          dict  = None   # current task definition
        self._turn:          int   = 0      # turns elapsed this episode
        self._history:       list  = []     # conversation log (strings)
        self._status:        str   = "open" # open|pending|resolved|closed
        self._classified:    bool  = False  # has the ticket been labelled?
        self._total_reward:  float = 0.0    # cumulative episode reward
        self._actions_log:   list  = []     # [{turn, action, reward, status}]

    # ---------------------------------------------------------------- #
    #  reset()                                                           #
    # ---------------------------------------------------------------- #
    def reset(self) -> Observation:
        """
        Start a fresh episode.

        Selects a task, zeroes all state, logs the opening customer
        message, and returns the first Observation the agent sees.

        Returns
        -------
        Observation
        """
        # ── 1. Select task ──────────────────────────────────────────
        if self.task_id:
            task = TASK_INDEX.get(self.task_id)
            if task is None:
                raise ValueError(
                    f"Unknown task_id '{self.task_id}'. "
                    f"Available: {list(TASK_INDEX.keys())}"
                )
        elif self.random_task:
            task = random.choice(TASKS)
        else:
            task = TASKS[0]

        # ── 2. Zero all internal state ───────────────────────────────
        self._task         = task
        self._turn         = 0
        self._status       = "open"
        self._classified   = False
        self._total_reward = 0.0
        self._actions_log  = []

        # ── 3. Seed the conversation with the customer's opening line ─
        self._history = [f"[CUSTOMER] {task['message']}"]

        return self._build_observation()

    # ---------------------------------------------------------------- #
    #  state (read-only property)                                        #
    # ---------------------------------------------------------------- #
    @property
    def state(self) -> dict:
        """
        Full internal state as a plain dictionary.

        Used by:
          - grader.py (calculate_reward, grade_episode)
          - API /state endpoint
          - tests
          - inference.py policy
        """
        if self._task is None:
            return {}

        return {
            # Task identity
            "task_id":      self._task["id"],
            "task_type":    self._task["type"],
            "difficulty":   self._task["difficulty"],
            "customer_msg": self._task["message"],

            # Progress tracking
            "turn":         self._turn,
            "max_turns":    self._task["max_turns"],
            "status":       self._status,
            "classified":   self._classified,

            # Logs
            "history":      list(self._history),
            "actions_log":  list(self._actions_log),

            # Reward
            "total_reward": round(self._total_reward, 4),
        }

    # ---------------------------------------------------------------- #
    #  step()  ← Day 2 core deliverable                                 #
    # ---------------------------------------------------------------- #
    def step(self, action: Action):
        """
        Advance the environment by one turn.

        Flow per step
        -------------
        1.  Validate env is ready
        2.  Increment turn counter
        3.  Apply step_penalty (every turn, tiny cost for efficiency)
        4.  Route to the correct action handler
        5.  Check for efficiency bonus
        6.  Check for max_turns termination
        7.  Accumulate total reward
        8.  Append to conversation history and actions_log
        9.  Cross-validate with deterministic grader
        10. Build and return (Observation, reward, done, info)

        Parameters
        ----------
        action : Action   pydantic model with action_type + content

        Returns
        -------
        observation : Observation   what the agent sees next
        reward      : float         reward for this step
        done        : bool          True = episode is over
        info        : dict          debug metadata
        """
        # ── Guard ────────────────────────────────────────────────────
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        # ── 1. Increment turn ────────────────────────────────────────
        self._turn += 1

        # ── 2. Base reward = step penalty (applied every single turn) ─
        reward = REWARD_CONFIG["step_penalty"]
        done   = False
        info   = {}

        atype   = action.action_type
        content = action.content.strip()

        # ── 3. Route to action handler ───────────────────────────────
        #       Each handler returns (delta_reward, info_fragment)
        if atype == "classify":
            delta, info = self._handle_classify(content)

        elif atype == "ask":
            delta, info = self._handle_ask(content)

        elif atype == "reply":
            delta, info = self._handle_reply(content)

        elif atype == "escalate":
            delta, info = self._handle_escalate(content)

        elif atype == "close":
            delta, info = self._handle_close(content)
            done = True   # close always ends the episode

        else:
            delta = REWARD_CONFIG["invalid_action"]
            info  = {"error": f"Unknown action_type '{atype}'"}

        reward += delta

        # ── 4. Efficiency bonus ──────────────────────────────────────
        #       If the ticket is resolved in under 60 % of max turns,
        #       award a bonus to encourage fast, accurate agents.
        max_t = self._task["max_turns"]
        if self._status == "resolved" and self._turn < max_t * 0.6:
            reward += REWARD_CONFIG["efficiency_bonus"]
            info["efficiency_bonus"] = True

        # ── 5. Max turns check ───────────────────────────────────────
        #       If we've hit the turn limit without closing, force done
        #       and apply a penalty for failing to resolve in time.
        if self._turn >= max_t and not done:
            done = True
            info["reason"] = "max_turns_reached"
            if self._status not in ("resolved", "closed"):
                reward += REWARD_CONFIG["close_wrong"]
                info["timeout_penalty"] = True

        # ── 6. Accumulate total reward ───────────────────────────────
        self._total_reward += reward

        # ── 7. Update conversation history ───────────────────────────
        self._history.append(
            f"[AGENT  t={self._turn}] {atype.upper()} | {content} "
            f"→ reward={reward:+.3f} | status={self._status}"
        )

        # ── 8. Update actions log ────────────────────────────────────
        self._actions_log.append({
            "turn":   self._turn,
            "action": atype,
            "reward": round(reward, 4),
            "status": self._status,
        })

        # ── 9. Cross-validate with grader ────────────────────────────
        info["grader_score"] = calculate_reward(self.state, atype)

        # ── 10. Build episode summary if done ─────────────────────────
        if done:
            info["episode"] = self._build_episode_summary()

        return self._build_observation(), round(reward, 4), done, info

    # ---------------------------------------------------------------- #
    #  Action handlers (private)                                         #
    # ---------------------------------------------------------------- #

    def _handle_classify(self, content: str):
        """
        classify — label the ticket type.

        Correct  : content matches task["type"] (case-insensitive)
                   → status becomes "pending", classified = True
        Wrong    : wrong label → small penalty, no state change
        Repeated : already classified → penalise waste
        """
        info = {}

        if self._classified:
            info["classify"] = "repeated — already classified"
            return REWARD_CONFIG["classify_repeat"], info

        expected = self._task["type"]
        if content.lower() == expected.lower():
            self._classified = True
            self._status     = "pending"          # ← status transition 1
            info["classify"] = f"correct ('{expected}')"
            return REWARD_CONFIG["classify_correct"], info
        else:
            info["classify"] = f"wrong (expected '{expected}', got '{content}')"
            return REWARD_CONFIG["classify_wrong"], info

    def _handle_ask(self, content: str):
        """
        ask — request clarification from the customer.

        Valid    : contains '?' and is at least 10 characters
                   → keeps status as "pending"
        Invalid  : too short or not a question → penalty
        """
        info = {}

        if len(content) < 10 or "?" not in content:
            info["ask"] = "invalid — must be a meaningful question containing '?'"
            return REWARD_CONFIG["ask_invalid"], info

        # Status stays "pending" — asking doesn't resolve the ticket
        if self._status == "open":
            self._status = "pending"              # ← status transition 1.5
        info["ask"] = "valid"
        return REWARD_CONFIG["ask_valid"], info

    def _handle_reply(self, content: str):
        """
        reply — provide the resolution to the customer.

        Correct  : content contains ≥ 1 keyword from task["reply_keywords"]
                   AND is at least 15 characters long
                   → status becomes "resolved"
        Wrong    : irrelevant content or too short → penalty
        """
        info = {}

        if len(content) < 15:
            info["reply"] = "too short to be a valid reply (< 15 chars)"
            return REWARD_CONFIG["reply_wrong"], info

        keywords = self._task.get("reply_keywords", [])
        matched  = [k for k in keywords if k in content.lower()]

        if matched:
            self._status    = "resolved"          # ← status transition 2
            info["reply"]   = f"correct — matched keywords: {matched}"
            return REWARD_CONFIG["reply_correct"], info
        else:
            info["reply"] = (
                f"no keyword match — expected one of {keywords}. "
                f"Reply was: '{content[:60]}...'"
            )
            return REWARD_CONFIG["reply_wrong"], info

    def _handle_escalate(self, content: str):
        """
        escalate — hand off to a senior support team.

        Correct  : task["escalate_allowed"] is True
                   → status becomes "resolved" (senior team handles it)
        Wrong    : escalating a simple ticket → bigger penalty
                   (wastes team resources on an easy issue)
        """
        info = {}

        if self._task.get("escalate_allowed", False):
            self._status       = "resolved"       # ← status transition 2b
            info["escalate"]   = "correct — this task warrants escalation"
            return REWARD_CONFIG["escalate_correct"], info
        else:
            info["escalate"] = (
                "wrong — escalation not needed for this task type. "
                f"Task type is '{self._task['type']}'"
            )
            return REWARD_CONFIG["escalate_wrong"], info

    def _handle_close(self, content: str):
        """
        close — mark the ticket as done and end the episode.

        Correct  : status is currently "resolved"
                   → status becomes "closed"
        Wrong    : status is not "resolved" → large penalty
                   (closing without solving = worst outcome)
        """
        info = {}

        if self._status == "resolved":
            self._status    = "closed"            # ← status transition 3
            info["close"]   = "correct — ticket resolved before closing"
            return REWARD_CONFIG["close_correct"], info
        else:
            info["close"] = (
                f"wrong — cannot close a ticket with status '{self._status}'. "
                "Ticket must be 'resolved' first."
            )
            return REWARD_CONFIG["close_wrong"], info

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #
    def _build_observation(self) -> Observation:
        """Build the Observation the agent receives."""
        # Map internal "resolved" to "pending" in the public Observation
        # so the agent only sees open / pending / closed in normal flow.
        # (resolved → closed happens in the next step when agent closes)
        obs_status = self._status
        if obs_status not in ("open", "pending", "resolved", "closed"):
            obs_status = "pending"

        return Observation(
            customer_message = self._task["message"],
            history          = list(self._history),
            status           = obs_status,
            turn             = self._turn,
            max_turns        = self._task["max_turns"],
            task_id          = self._task["id"],
            task_type        = self._task["type"],
        )

    def _build_episode_summary(self) -> dict:
        """Build the end-of-episode report included in info."""
        summary = {
            "task_id":      self._task["id"],
            "task_type":    self._task["type"],
            "total_reward": round(self._total_reward, 4),
            "turns_used":   self._turn,
            "max_turns":    self._task["max_turns"],
            "final_status": self._status,
            "success":      self._status in ("resolved", "closed"),
        }
        summary["grade"] = grade_episode(summary)
        return summary