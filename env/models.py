from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Action(BaseModel):
    """
    The agent's chosen action for one turn.

    action_type : what the agent wants to do
    content     : the actual message, label, or question text
    """
    action_type: Literal["classify", "ask", "reply", "escalate", "close"]
    content: str = Field(..., description="Message or label based on action type")


class HistoryItem(BaseModel):
    """One entry in the conversation log."""
    turn:   int
    role:   Literal["customer", "agent"]
    action: str
    text:   str
    reward: float


class Observation(BaseModel):
    """
    Everything the agent can see at the start of each turn.

    This is the agent's full information window — it sees
    the customer message, the full conversation history,
    the current ticket status, and the turn counters.
    """
    customer_message: str
    history:          List[str]          # human-readable log lines
    status:           Literal["open", "pending", "resolved", "closed"]
    turn:             int
    max_turns:        int
    task_id:          Optional[str] = None
    task_type:        Optional[str] = None


class StepResult(BaseModel):
    """Return value of env.step() in structured form."""
    observation: Observation
    reward:      float
    done:        bool
    info:        dict


class EpisodeSummary(BaseModel):
    """Final report produced when an episode ends."""
    task_id:      str
    task_type:    str
    total_reward: float
    turns_used:   int
    max_turns:    int
    final_status: str
    success:      bool