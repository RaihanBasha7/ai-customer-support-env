from fastapi import FastAPI
import uvicorn
from env.environment import CustomerSupportEnv

app = FastAPI()
env = CustomerSupportEnv()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    state, reward, done, _ = env.step(action)
    return {
        "state": state,
        "reward": round(float(reward), 2),
        "done": bool(done)
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()