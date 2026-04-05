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
    
        # ❌ INVALID ACTION
        if action_type not in valid_actions:
            reward = REWARD_CONFIG["invalid_action"]
            info["error"] = "invalid_action"
            reason = "Invalid action"
    
        # -------------------------
        # ENFORCE ACTION SEQUENCE 🔥
        # -------------------------
        last_action = self.state.get("last_action")
    
        if last_action == "classify" and action_type == "classify":
            reward -= 0.2
            reason = "Repeated classification"
    
        if last_action == "reply" and action_type == "ask":
            reward -= 0.2
            reason = "Unnecessary ask after reply"
    
        # -------------------------
        # CLASSIFY (SIMPLIFIED + STRONG)
        # -------------------------
        if action_type == "classify":
            if task_type in content:
                reward = REWARD_CONFIG["classify_correct"]
                self.state["classification"] = task_type
                reason = "Correct classification"
            else:
                reward = REWARD_CONFIG["classify_wrong"]
                reason = f"Expected: {task_type}"
    
        # -------------------------
        # ASK
        # -------------------------
        elif action_type == "ask":
            if self.state["classification"] is None:
                reward = REWARD_CONFIG["ask_valid"]
                self.state["status"] = "pending"
                reason = "Valid clarification"
            else:
                reward = REWARD_CONFIG["ask_valid"] - 0.1
                reason = "Redundant ask"
    
        # -------------------------
        # REPLY
        # -------------------------
        elif action_type == "reply":
        
            if self.state["classification"] != task_type:
                reward = -0.3
                reason = "Reply without correct classification"
    
            elif task_type == "refund" and "refund" in content:
                reward = REWARD_CONFIG["reply_correct"]
                reason = "Refund handled"
    
            elif task_type == "angry" and ("sorry" in content):
                reward = REWARD_CONFIG["reply_correct"]
                reason = "Empathy shown"
    
            elif task_type == "complex":
                if "refund" in content and "compensation" in content:
                    reward = REWARD_CONFIG["reply_correct"]
                    reason = "Handled complex issue"
                else:
                    reward = REWARD_CONFIG["reply_wrong"]
                    reason = "Incomplete response"
    
            elif task_type == "edge_case":
                reward = REWARD_CONFIG["reply_wrong"]
                reason = "Should escalate"
    
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
                reason = "Correct escalation"
            else:
                reward = REWARD_CONFIG["escalate_wrong"]
                reason = "Unnecessary escalation"
    
        # -------------------------
        # CLOSE (IMPROVED 🔥)
        # -------------------------
        elif action_type == "close":
            if (
                self.state.get("classification") == task_type
                and last_action in ["reply", "escalate"]
            ):
                reward = REWARD_CONFIG["close_correct"]
                self.state["status"] = "closed"
                done = True
                reason = "Proper closure"
            else:
                reward = REWARD_CONFIG["close_wrong"]
                reason = "Invalid closing"
    
        # -------------------------
        # STEP LIMIT
        # -------------------------
        if self.turn >= self.current_task["max_turns"]:
            reward += REWARD_CONFIG["step_penalty"]
            done = True
    
        # -------------------------
        # LIGHT PENALTY (SMART)
        # -------------------------
        reward -= 0.02 * self.turn
    
        # -------------------------
        # HISTORY (EXPLAINABILITY 🔥)
        # -------------------------
        self.state["conversation"].append(
            f"Step {self.turn}: {action_type} → reward={round(reward,2)} | {reason}"
        )
    
        self.state["last_action"] = action_type
    
        # -------------------------
        # SMART HINTS
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
                task_type,
                self.current_task["max_turns"]
            )
            info["final_score"] = final_score
    
        return {
            "customer_message": self.current_task["message"],
            "conversation": self.state["conversation"],
            "status": self.state["status"]
        }, round(reward, 2), done, info