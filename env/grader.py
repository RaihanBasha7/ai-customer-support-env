def calculate_reward(state, action):
    score = 0.0

    # basic logic
    if action == "close" and state["status"] == "resolved":
        score += 1.0

    return min(score, 1.0)