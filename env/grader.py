def compute_score(state, task_type, max_turns):
    score = 0.0
    logs = []

    conversation = state.get("conversation", [])
    classification = state.get("classification")
    status = state.get("status")

    steps = len(conversation)

    # Classification
    if classification == task_type:
        score += 0.3
        logs.append("Correct classification")
    else:
        logs.append("Wrong classification")

    # Resolution
    if status == "closed":
        score += 0.4
        logs.append("Resolved successfully")
    else:
        logs.append("Not resolved")

    # Efficiency
    if steps <= max_turns:
        score += 0.3
        logs.append("Efficient steps")
    else:
        penalty = 0.05 * (steps - max_turns)
        score += max(0, 0.3 - penalty)
        logs.append("Too many steps")

    return {
        "final_score": round(min(score, 1.0), 2),
        "logs": logs
    }