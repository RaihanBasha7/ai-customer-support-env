import os
import json
from env.environment import CustomerSupportEnv
from openai import OpenAI
import os
import json


client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def call_llm(observation):
    history = observation["conversation"]
    user_msg = observation["customer_message"]

    prompt = f"""
You are a customer support AI agent.

Decide the next action based on the conversation.
Be efficient. Solve in minimum steps. Be polite.

Available actions:
- classify
- ask
- reply
- escalate
- close

Return ONLY JSON:
{{
  "action_type": "...",
  "content": "..."
}}

Customer message:
{user_msg}

Conversation:
{history}
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": "You are a smart support agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        text = response.choices[0].message.content.strip()

        return json.loads(text)

    except Exception as e:
        # fallback (VERY IMPORTANT)
        return {"action_type": "ask", "content": "fallback"}
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