from pydantic import BaseModel, Field
from typing import List, Literal


class Action(BaseModel):
    action_type: Literal["classify", "ask", "reply", "escalate", "close"]
    content: str = Field(..., description="Message or label based on action")


class HistoryItem(BaseModel):
    step: int
    action: str
    reward: float


class Observation(BaseModel):
    customer_message: str
    history: List[str]
    status: Literal["open", "pending", "closed"]
    turn: int
    max_turns: int