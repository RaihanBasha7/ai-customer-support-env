import os
import json
from openai import OpenAI
from env.environment import CustomerSupportEnv
# Verify env vars on startup — crashes early with a clear message if missing
assert os.environ.get("API_BASE_URL"), "ERROR: API_BASE_URL is not set!"
assert os.environ.get("API_KEY"), "ERROR: API_KEY is not set!"
print(f"[ENV OK] API_BASE_URL = {os.environ['API_BASE_URL']}")
print(f"[ENV OK] API_KEY = {os.environ['API_KEY'][:8]}...")  # only show first 8 chars

# ─────────────────────────────────────────────
# USE THE HACKATHON'S INJECTED PROXY — REQUIRED
# ─────────────────────────────────────────────
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.environ.get("API_KEY", ""),
)

# Use plain model name — their LiteLLM proxy maps this internally
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """You are an AI customer support agent. 
At each step you must respond with a single JSON object — nothing else.

The JSON must have exactly two keys:
  "action_type": one of ["classify", "ask", "reply", "escalate", "close"]
  "content": a string (the message or label for that action)

Action rules:
1. Always start by classifying the issue with action_type="classify".
   Use one of these labels: refund, angry, complex, edge_case
2. Then reply to the customer with action_type="reply".
   - For refund: confirm refund has been initiated.
   - For angry: show empathy — use words like "sorry" or "apologize".
   - For complex: address both the refund and compensation.
   - For edge_case: DO NOT reply — escalate immediately.
3. For edge_case or complex issues, use action_type="escalate".
4. End every resolved session with action_type="close" and content="".

IMPORTANT: Output ONLY valid JSON. No explanation, no markdown, no extra text.
"""


def call_llm(observation: dict) -> dict:
    """Call the LiteLLM proxy and return a structured action."""
    conversation = observation.get("conversation", [])
    customer_message = observation.get("customer_message", "")

    # Build a human-readable conversation summary for the model
    history_text = "\n".join(conversation) if conversation else "None yet."

    user_prompt = f"""Customer message: {customer_message}

Conversation so far:
{history_text}

What is your next action? Respond ONLY with valid JSON."""

    response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.2,
    max_tokens=150,
)

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    action = json.loads(raw)

    # Validate required keys
    assert "action_type" in action, "Missing action_type"
    assert "content" in action, "Missing content"

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
            print(f"[LLM ERROR] {e} — falling back to close")
            action = {"action_type": "close", "content": ""}

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

