from fastapi import FastAPI
from env.environment import CustomerSupportEnv

app = FastAPI()
env = CustomerSupportEnv()

@app.get("/")
def root():
    return {"message": "Customer Support Env Running"}

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    state, reward, done, info = env.step(action)

    return {
        "state": state,
        "reward": reward,
        "done": done,

        # 🔥 clean output for UI/judges
        "final_score": info.get("final_score"),
        "logs": info.get("grading_logs", []),
        "reason": info.get("reason", ""),
        
        # optional debug info
        "info": info
    }