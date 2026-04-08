REWARD_CONFIG = {
    # ── Classification ────────────────────────────────────────────
    "classify_correct":  0.30,   # right label on first try
    "classify_wrong":   -0.20,   # wrong label
    "classify_repeat":  -0.10,   # classified twice (wasteful)

    # ── Clarifying questions ──────────────────────────────────────
    "ask_valid":         0.10,   # proper question with '?'
    "ask_invalid":      -0.10,   # not a proper question

    # ── Customer reply ────────────────────────────────────────────
    "reply_correct":     0.50,   # relevant keywords matched → resolved
    "reply_wrong":      -0.30,   # irrelevant or too short

    # ── Escalation ────────────────────────────────────────────────
    "escalate_correct":  0.30,   # escalated a complex/edge_case ticket
    "escalate_wrong":   -0.40,   # escalated a simple ticket (unnecessary)

    # ── Closing ───────────────────────────────────────────────────
    "close_correct":     0.20,   # closed after resolved
    "close_wrong":      -0.50,   # closed before resolved (worst mistake)

    # ── Global penalties / bonuses ────────────────────────────────
    "invalid_action":   -0.50,   # unrecognised action_type
    "step_penalty":     -0.02,   # small per-turn cost (encourages efficiency)
    "efficiency_bonus":  0.20,   # resolved in < 60 % of max_turns
}

# Grade thresholds (used by grader.py)
GRADE_THRESHOLDS = {
    "excellent": 0.85,
    "good":      0.60,
    "partial":   0.30,
    "fail":      0.00,
}