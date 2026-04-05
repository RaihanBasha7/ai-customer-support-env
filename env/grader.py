from .reward_config import REWARD_CONFIG, GRADE_THRESHOLDS

def calculate_reward(state: dict, action: str) -> float:
    """
    Cross-validate the inline reward produced by the environment.
    Operates on the state snapshot AFTER the action was applied.

    Parameters
    ----------
    state  : dict   — env.state snapshot
    action : str    — action_type string

    Returns
    -------
    float in [0.0, 1.0]
    """
    score        = 0.0
    status       = state.get("status", "open")
    classified   = state.get("classified", False)
    turn         = state.get("turn", 0)
    max_turns    = state.get("max_turns", 1)
    task_type    = state.get("task_type", "")
    escalatable  = task_type in ("complex", "edge_case")
    resolved     = status in ("resolved", "closed")

    if action == "classify":
        # Full credit if classified correctly, none if not
        score = 0.5 if classified else 0.0

    elif action == "ask":
        # More valuable early in complex tasks
        if escalatable and turn <= 2:
            score = 0.35
        else:
            score = 0.10

    elif action == "reply":
        if resolved:
            # Reward speed — faster resolution = higher grader score
            efficiency = 1.0 - (turn / max(max_turns, 1))
            score = round(0.5 + 0.4 * efficiency, 4)   # range: 0.5 – 0.9
        else:
            score = 0.10   # partial credit for attempting

    elif action == "escalate":
        if escalatable and resolved:
            score = 0.65
        elif escalatable:
            score = 0.30
        else:
            score = 0.0    # escalating easy ticket = wrong decision

    elif action == "close":
        if resolved:
            efficiency = 1.0 - (turn / max(max_turns, 1))
            score = round(0.7 + 0.3 * efficiency, 4)   # range: 0.7 – 1.0
        else:
            score = 0.0    # closing unresolved = hard failure

    return round(min(max(score, 0.0), 1.0), 4)


# ──────────────────────────────────────────────────────────────────────
#  End-of-episode grader
# ──────────────────────────────────────────────────────────────────────
def grade_episode(summary: dict) -> dict:
    """
    Produce a structured grade report for a completed episode.

    Parameters
    ----------
    summary : dict with keys:
        total_reward, turns_used, max_turns, final_status, task_type

    Returns
    -------
    dict:
        normalised_score : float  — [0.0, 1.0]
        grade            : str    — excellent / good / partial / fail
        success          : bool   — ticket resolved or closed
        efficiency       : float  — 1 - turns_used/max_turns
        passed           : bool   — grade in (excellent, good)
    """
    raw          = summary.get("total_reward", 0.0)
    final_status = summary.get("final_status", "open")
    turns_used   = summary.get("turns_used", 1)
    max_turns    = summary.get("max_turns", 1)

    # Max achievable reward in a perfect episode
    max_possible = (
        REWARD_CONFIG["classify_correct"]
        + REWARD_CONFIG["reply_correct"]
        + REWARD_CONFIG["close_correct"]
        + REWARD_CONFIG["efficiency_bonus"]
    )

    normalised = round(min(max(raw / max(max_possible, 1), 0.0), 1.0), 4)

    if normalised >= GRADE_THRESHOLDS["excellent"]:
        grade = "excellent"
    elif normalised >= GRADE_THRESHOLDS["good"]:
        grade = "good"
    elif normalised >= GRADE_THRESHOLDS["partial"]:
        grade = "partial"
    else:
        grade = "fail"

    return {
        "normalised_score": normalised,
        "grade":            grade,
        "success":          final_status in ("resolved", "closed"),
        "efficiency":       round(1.0 - turns_used / max(max_turns, 1), 4),
        "passed":           grade in ("excellent", "good"),
    }