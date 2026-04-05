from .models import Action, Observation
from .tasks import TASKS
from .reward_config import REWARD_CONFIG
from .grader import compute_score
import random

class CustomerSupportEnv:
    def __init__(self):
        self.state = None
        self.current_task = None
        self.turn = 0

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.current_task = random.choice(TASKS)  # :contentReference[oaicite:0]{index=0}

        self.state = {
    "conversation": [],
    "status": "open",
    "classification": None  
}

        self.turn = 0

        return {
            "customer_message": self.current_task["message"],
            "conversation": self.state["conversation"],
            "status": self.state["status"]
        }

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):
        reward = 0.0
        done = False
        info = {}
        reason = ""

        self.turn += 1
        action_type = action.get("action_type")
        content = action.get("content", "").lower()

        task_type = self.current_task["type"]

        valid_actions = ["classify", "ask", "reply", "escalate", "close"]

        # ❌ Invalid action
        if action_type not in valid_actions:
            reward = REWARD_CONFIG["invalid_action"]  # :contentReference[oaicite:1]{index=1}
            info["error"] = "invalid_action"

        # -------------------------
        # CLASSIFY
        # -------------------------
        elif action_type == "classify":

            keywords = {
                "refund": ["refund", "return"],
                "angry": ["terrible", "bad", "worst", "angry"],
                "complex": ["delay", "compensation"],
                "edge_case": ["fraud", "charged", "unauthorized"]
            }

            if task_type == "complex":
                if "refund" in content and "compensation" in content:
                    reward = REWARD_CONFIG["classify_correct"]
                    self.state["classification"] = task_type
                    reason = "Correct complex classification"
                else:
                    reward = REWARD_CONFIG["classify_wrong"]
                    reason = "Incorrect classification for complex issue"

            elif any(word in content for word in keywords.get(task_type, [])):
                reward = REWARD_CONFIG["classify_correct"]
                self.state["classification"] = task_type
                reason = f"Detected keyword for {task_type}"

            else:
                reward = REWARD_CONFIG["classify_wrong"]
                reason = "Incorrect classification"
                info["hint"] = f"Expected: {task_type}"
        # -------------------------
        # ASK
        # -------------------------
        elif action_type == "ask":
            reward = REWARD_CONFIG["ask_valid"]
            self.state["status"] = "pending"

        # -------------------------
        # REPLY
        # -------------------------
        elif action_type == "reply":

            if task_type == "refund" and "refund" in content:
                reward = REWARD_CONFIG["reply_correct"]
                reason = "Proper refund response"

            elif task_type == "angry" and ("sorry" in content or "apologize" in content):
                reward = REWARD_CONFIG["reply_correct"]
                reason = "Empathetic response"

            elif task_type == "complex":
                if "refund" in content and "compensation" in content:
                    reward = REWARD_CONFIG["reply_correct"]
                    reason = "Handled both refund and compensation"
                else:
                    reward = REWARD_CONFIG["reply_wrong"]
                    reason = "Incomplete resolution"

            elif task_type == "edge_case":
                reward = REWARD_CONFIG["reply_wrong"]
                reason = "Should escalate instead of replying"

            else:
                reward = REWARD_CONFIG["reply_wrong"]
                reason = "Incorrect reply"

        # -------------------------
        # ESCALATE
        # -------------------------
        elif action_type == "escalate":

            if task_type in ["complex", "edge_case"]:
                reward = REWARD_CONFIG["escalate_correct"]
                self.state["status"] = "closed"
                done = True
                reason = f"Escalated {task_type} issue appropriately"
            else:
                reward = REWARD_CONFIG["escalate_wrong"]
                reason = "Unnecessary escalation"
        
        # -------------------------
        # CLOSE
        # -------------------------
        elif action_type == "close":
            if (
                self.state.get("classification") == task_type
                and self.turn > 1
            ):
                reward = REWARD_CONFIG["close_correct"]
                self.state["status"] = "closed"
                done = True
                reason = "Issue correctly resolved and closed"
            else:
                reward = REWARD_CONFIG["close_wrong"]
                reason = "Premature or incorrect closing"

        # -------------------------
        # STEP LIMIT
        # -------------------------
        if self.turn >= self.current_task["max_turns"]:
            # reward -= 0.05 * self.turn
            done = True
        # -------------------------
        # UPDATE HISTORY
        # -------------------------
        # apply penalty FIRST
        reward -= 0.05 * self.turn

        # then log
        self.state["conversation"].append(
            f"Step {self.turn}: {action_type} → reward={round(reward,2)} | {reason}"
        )
        
        # -------------------------
        # MEMORY (NEW)
        # -------------------------
        self.state["last_action"] = action_type

        # -------------------------
        # SUGGESTION (NEW)
        # -------------------------
        if not done:
            if task_type == "complex":
                info["next_best_action"] = "reply or escalate"
            elif task_type == "edge_case":
                info["next_best_action"] = "escalate"

        # -------------------------
        # FINAL SCORE
        # -------------------------
        if done:
            final_score = compute_score(
                self.state,
                self.current_task["type"],
                self.current_task["max_turns"]
            )
            info["final_score"] = final_score
        reward = round(reward, 2)
        return {
            "customer_message": self.current_task["message"],
            "conversation": self.state["conversation"],
            "status": self.state["status"]
        }, reward, done, info