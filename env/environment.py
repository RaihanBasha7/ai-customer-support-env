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
    def reset(self, task_id=None):
        if task_id is not None:
            self.current_task = TASKS[task_id]
        else:
            self.current_task = TASKS[0]  # default deterministic # :contentReference[oaicite:0]{index=0}

        self.state = {
    "conversation": [],
    "status": "open",
    "classification": None,
    "resolved": False,
    "confidence": 0.0
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
            content = content.lower()

            intent_keywords = {
    "refund": ["refund", "return", "money", "charged", "payment"],
    "angry": ["angry", "not happy", "bad service", "terrible", "worst"],
    "complex": ["issue", "problem", "multiple", "delay", "compensation"],
    "edge_case": ["fraud", "hack", "unauthorized", "unknown charge"]
}

            matched = False

            for intent, keywords in intent_keywords.items():
                if any(word in content for word in keywords):
                    if intent == task_type:
                        reward = REWARD_CONFIG["classify_correct"] + 0.1
                        self.state["classification"] = intent
                        reason = "Correct classification"
                    else:
                        reward = REWARD_CONFIG["classify_wrong"]
                        reason = f"Wrong classification (expected {task_type})"
                    matched = True
                    break
                
            if not matched:
                reward = REWARD_CONFIG["classify_wrong"]
                reason = "Could not understand intent"

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

            content = content.lower()
        
            if self.state["classification"] != task_type:
                reward = -0.3
                reason = "Reply without correct classification"
        
            elif task_type == "refund":
                if any(word in content for word in ["refund", "processed", "initiated", "return"]):
                    reward = REWARD_CONFIG["reply_correct"]
                    reason = "Refund handled"
                else:
                    reward = REWARD_CONFIG["reply_wrong"]
                    reason = "Refund not handled properly"
        
            elif task_type == "angry":
                if any(word in content for word in ["sorry", "apologize", "understand"]):
                    reward = REWARD_CONFIG["reply_correct"]
                    reason = "Empathy shown"
                else:
                    reward = REWARD_CONFIG["reply_wrong"]
                    reason = "No empathy"
        
            elif task_type == "complex":
                if any(word in content for word in ["refund", "compensation", "resolve"]):
                    reward = REWARD_CONFIG["reply_correct"]
                    reason = "Handled complex issue"
                else:
                    reward = REWARD_CONFIG["reply_wrong"]
                    reason = "Incomplete response"
        
            elif task_type == "edge_case":
                reward = REWARD_CONFIG["reply_wrong"]
                reason = "Should escalate"

            # bonus for polite tone
            elif any(word in content for word in ["sorry", "apologize", "understand"]):
                reward += 0.1
    
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

            #❌ Missing classification
            if self.state.get("classification") is None:
                reward = REWARD_CONFIG["close_wrong"] - 0.5
                reason = "Closed without classification"

            # ❌ Premature close (no proper resolution step)
            elif last_action not in ["reply", "escalate"]:
                reward = REWARD_CONFIG["close_wrong"] - 0.3
                reason = "Premature closing"

            # ✅ Proper close
            elif self.state.get("classification") == task_type:
                reward = REWARD_CONFIG["close_correct"]
                self.state["status"] = "closed"
                done = True
                reason = "Proper closure"

            # ❌ Wrong classification but closed
            else:
                reward = REWARD_CONFIG["close_wrong"] - 0.2
                reason = "Closed with wrong classification"
    
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
        # -------------------------
# FINAL SCORE
# -------------------------
        if done:
            grading = compute_score(
                self.state,
                task_type,
                self.current_task["max_turns"]
            )

            info["final_score"] = grading["final_score"]
            info["grading_logs"] = grading.get("logs", [])
        else:
            info["final_score"] = None
            info["grading_logs"] = []

        info["reason"] = reason
        return {
            "customer_message": self.current_task["message"],
            "conversation": self.state["conversation"],
            "status": self.state["status"]
        }, round(reward, 2), done, info