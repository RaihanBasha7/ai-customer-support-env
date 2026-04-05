import os
import json
import requests
from env.environment import CustomerSupportEnv
import google.generativeai as genai
import os
import json
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

def call_llm(observation):
    prompt = f"""
You are an AI agent in a structured environment.

STRICT RULES:
- You MUST follow this sequence:
  classify → reply → close
- DO NOT repeat actions
- DO NOT ask unnecessary questions
- Output ONLY JSON

Format:
{{"action_type": "...", "content": "..."}}

Allowed actions:
classify, ask, reply, escalate, close

Task:
Customer message: {observation["customer_message"]}
History: {observation["conversation"]}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text

        # Extract JSON
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            return json.loads(match.group())

    except Exception as e:
        print("Gemini Error:", e)

    return {"action_type": "ask", "content": "Fallback"}
def run_episode():
    env = CustomerSupportEnv()
    obs = env.reset()

    print("[START]")
    print(obs)

    done = False

    while not done:
        action = call_llm(obs)

        print("\n[STEP]")
        print("Action:", action)

        obs, reward, done, info = env.step(action)

        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

    print("\n[END]")


if __name__ == "__main__":
    run_episode()