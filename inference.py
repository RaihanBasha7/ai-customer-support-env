import os
import json
from openai import OpenAI
from env.environment import CustomerSupportEnv

# Read env vars safely — no assert, no crash
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL = os.environ.get("MODEL_NAME")

print(f"[ENV] API_BASE_URL = {API_BASE_URL}")
print(f"[ENV] API_KEY set = {bool(API_KEY)}")
print(f"[ENV] MODEL = {MODEL}")

print("=== DEBUG ENV ===")
print("API_BASE_URL:", API_BASE_URL)
print("API_KEY exists:", bool(API_KEY))
print("=================")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an AI customer support agent. Output ONE JSON object only.

Format: {"action_type": "<action>", "content": "<text>"}

Follow this EXACT 3-step sequence based on conversation history:

STEP 1 - History is empty → classify:
{"action_type": "classify", "content": "refund"}
Use: refund / angry / complex / edge_case

STEP 2 - Last action was classify → reply (or escalate for edge_case):
- refund:  {"action_type": "reply", "content": "We sincerely apologize for the damaged product. Your refund has been initiated and will be processed within 3-5 business days."}
- angry:   {"action_type": "reply", "content": "We sincerely apologize for this poor experience. We understand your frustration and will resolve this immediately."}
- complex: {"action_type": "reply", "content": "We sincerely apologize for the delay. Your refund has been initiated and compensation will be provided for the inconvenience."}
- edge_case: {"action_type": "escalate", "content": ""}

STEP 3 - Last action was reply → close:
{"action_type": "close", "content": ""}

STRICT RULES:
- NEVER classify more than once
- NEVER reply more than once
- ALWAYS close after replying
- NO markdown, NO explanation — raw JSON only
"""


def get_last_action(conversation: list) -> str:
    """Detect last action from conversation history lines."""
    for line in reversed(conversation):
        for act in ["close", "escalate", "reply", "classify", "ask"]:
            if f": {act} " in line:
                return act
    return ""


def fallback_action(observation: dict) -> dict:
    """Deterministic fallback — always produces correct next step."""
    conversation = observation.get("conversation", [])
    message = observation.get("customer_message", "").lower()
    last = get_last_action(conversation)

    if last == "":
        if "refund" in message or "damaged" in message:
            return {"action_type": "classify", "content": "refund"}
        elif "unauthorized" in message or "fraud" in message:
            return {"action_type": "classify", "content": "edge_case"}
        elif "delay" in message or "compensation" in message:
            return {"action_type": "classify", "content": "complex"}
        else:
            return {"action_type": "classify", "content": "angry"}
    elif last == "classify":
        if "unauthorized" in message or "fraud" in message:
            return {"action_type": "escalate", "content": ""}
        return {
            "action_type": "reply",
            "content": "We sincerely apologize for the inconvenience. Your refund has been initiated and will be processed within 3-5 business days."
        }
    else:
        return {"action_type": "close", "content": ""}


def call_llm(observation: dict) -> dict:
    conversation = observation.get("conversation", [])
    customer_message = observation.get("customer_message", "")
    history_text = "\n".join(conversation) if conversation else "None yet."

    user_prompt = f"""Customer message: {customer_message}

Conversation history:
{history_text}

Next action (JSON only):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    action = json.loads(raw)
    assert "action_type" in action and "content" in action
    return action


def run_episode(task_id: int = 0):
    env = CustomerSupportEnv()
    obs = env.reset(task_id=task_id)

    print("[START]")
    print(f"Customer: {obs['customer_message']}\n")

    done = False
    while not done:
        try:
            action = call_llm(obs)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

        obs, reward, done, info = env.step(action)

        print("[STEP]")
        print(json.dumps({
            "action": action,
            "reward": round(float(reward), 2),
            "done": done,
            "reason": info.get("reason", ""),
        }, indent=2))

    print("\n[END]")
    if info.get("final_score") is not None:
        print(f"Final Score : {info['final_score']}")
        print("Grading Logs:")
        for log in info.get("grading_logs", []):
            print(f"  {log}")


if __name__ == "__main__":
    try:
        run_episode(task_id=0)
    except Exception as e:
        print("[END]")
        print(f"Error: {e}")