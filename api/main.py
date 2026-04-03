from fastapi import FastAPI
from env.environment import CustomerSupportEnv

app = FastAPI()
env = CustomerSupportEnv()

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: str):
    state, reward, done, _ = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }