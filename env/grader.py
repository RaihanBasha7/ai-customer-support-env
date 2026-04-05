def compute_score(state, task_type, max_turns):
    score = 0.0

    conversation = state.get("conversation", [])
    classification = state.get("classification")
    status = state.get("status")

    steps = len(conversation)

    # -------------------------
    # 1. CLASSIFICATION (0.3)
    # -------------------------
    if classification == task_type:
        score += 0.3

    # -------------------------
    # 2. RESOLUTION (0.4)
    # -------------------------
    if task_type in ["complex", "edge_case"]:
        # escalation or closure is valid
        if status == "closed":
            score += 0.4
    else:
        if status == "closed":
            score += 0.4

    # -------------------------
    # 3. EFFICIENCY (0.3)
    # -------------------------
    if steps <= max_turns:
        score += 0.3
    else:
        score += max(0, 0.3 - 0.05 * (steps - max_turns))

    return round(min(score, 1.0), 2)