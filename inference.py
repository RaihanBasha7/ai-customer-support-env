import os
import json
import requests
from env.environment import CustomerSupportEnv
import os
import json

def call_llm(observation):
    history = observation["conversation"]
    msg = observation["customer_message"].lower()

    if len(history) == 0:
        if "refund" in msg:
            return {"action_type": "classify", "content": "refund"}
        elif "unauthorized" in msg:
            return {"action_type": "classify", "content": "edge_case"}
        else:
            return {"action_type": "classify", "content": "complex"}

    elif len(history) == 1:
        if "refund" in msg:
            return {"action_type": "reply", "content": "Your refund has been initiated."}
        elif "unauthorized" in msg:
            return {"action_type": "escalate", "content": ""}
        else:
            return {"action_type": "reply", "content": "We are resolving your issue."}

    else:
        return {"action_type": "close", "content": ""}
def run_episode():
    env = CustomerSupportEnv()
    obs = env.reset()

    print("[START]")

    done = False

    while not done:
        action = call_llm(obs)

        obs, reward, done, info = env.step(action)

        print("[STEP]")
        print({
            "action": action,
            "reward": round(reward, 2),
            "done": str(done).lower()
        })

    print("[END]")

if __name__ == "__main__":
    try:
        run_episode()
    except Exception as e:
        print("[END]")
        print("Error:", str(e))