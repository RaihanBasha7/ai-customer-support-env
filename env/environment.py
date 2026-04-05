from .models import Action, Observation
from .tasks import TASKS
from .reward_config import REWARD_CONFIG
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
            "status": "open"
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
            if content == task_type:
                reward = REWARD_CONFIG["classify_correct"]
            else:
                reward = REWARD_CONFIG["classify_wrong"]

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
            if task_type in content:
                reward = REWARD_CONFIG["reply_correct"]
            else:
                reward = REWARD_CONFIG["reply_wrong"]

        # -------------------------
        # ESCALATE
        # -------------------------
        elif action_type == "escalate":
            if task_type in ["complex", "edge_case"]:
                reward = REWARD_CONFIG["escalate_correct"]
                done = True
            else:
                reward = REWARD_CONFIG["escalate_wrong"]

        # -------------------------
        # CLOSE
        # -------------------------
        elif action_type == "close":
            reward = REWARD_CONFIG["close_correct"]
            self.state["status"] = "closed"
            done = True

        # -------------------------
        # STEP LIMIT
        # -------------------------
        if self.turn >= self.current_task["max_turns"]:
            reward += REWARD_CONFIG["step_penalty"]
            done = True

        # -------------------------
        # UPDATE HISTORY
        # -------------------------
        self.state["conversation"].append(
            f"Step {self.turn}: {action_type} ({reward})"
        )

        return {
            "customer_message": self.current_task["message"],
            "conversation": self.state["conversation"],
            "status": self.state["status"]
        }, reward, done, info