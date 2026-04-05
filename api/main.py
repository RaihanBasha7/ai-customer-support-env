"""
main.py — OpenEnv FastAPI server

Endpoints:
  POST /reset        — start a new episode
  POST /step         — take one action
  GET  /state        — inspect current state
  GET  /tasks        — list all available tasks
  GET  /health       — liveness probe
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import CustomerSupportEnv, Action, TASKS
from env.grader import grade_episode

app = FastAPI(
    title="OpenEnv — AI Customer Support Training Environment",
    description=(
        "A reinforcement learning environment where an AI agent "
        "resolves customer support tickets step-by-step."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per session)
_env = CustomerSupportEnv()


# ── Request / Response schemas ────────────────────────────────────── #

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    random_task: bool = True


class StepRequest(BaseModel):
    action_type: str
    content: str


class ResetResponse(BaseModel):
    observation: dict
    state: dict
    message: str


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict
    state: dict


# ── Endpoints ─────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {"status": "ok", "service": "OpenEnv"}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks."""
    return {
        "tasks": [
            {
                "id":         t["id"],
                "difficulty": t["difficulty"],
                "type":       t["type"],
                "message":    t["message"],
                "max_turns":  t["max_turns"],
                "description": t["description"],
            }
            for t in TASKS
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    Body (optional):
        task_id     : str   — pick a specific task
        random_task : bool  — randomise task (default true)
    """
    _env.task_id     = req.task_id
    _env.random_task = req.random_task

    try:
        obs = _env.reset()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ResetResponse(
        observation = obs.model_dump(),
        state       = _env.state,
        message     = f"Episode started — task: {_env.state['task_id']}",
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Take one action in the environment.

    Body:
        action_type : classify | ask | reply | escalate | close
        content     : str  — message or label
    """
    if _env._task is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first."
        )

    valid_actions = ["classify", "ask", "reply", "escalate", "close"]
    if req.action_type not in valid_actions:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{req.action_type}'. "
                   f"Must be one of {valid_actions}."
        )

    action = Action(action_type=req.action_type, content=req.content)

    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation = obs.model_dump(),
        reward      = reward,
        done        = done,
        info        = info,
        state       = _env.state,
    )


@app.get("/state")
def get_state():
    """Return the current internal state."""
    if _env._task is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first."
        )
    return _env.state


# ── Entry point ───────────────────────────────────────────────────── #

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)