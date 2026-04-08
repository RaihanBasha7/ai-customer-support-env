def compute_score(state, task_type, max_turns):
    score = 0.0
    logs = []

    conversation = state.get("conversation", [])
    classification = state.get("classification")
    status = state.get("status")

    steps = len(conversation)
    convo_text = " ".join(conversation).lower()

    # -------------------------
    # 1. Classification (STRICT)
    # -------------------------
    if classification == task_type:
        score += 0.25
        logs.append("✅ Correct classification")
    else:
        score -= 0.1
        logs.append("❌ Wrong classification")

    # -------------------------
    # 2. Resolution Quality
    # -------------------------
    if status == "closed":
        score += 0.25
        logs.append("✅ Issue resolved")
    else:
        score -= 0.1
        logs.append("❌ Not resolved")

    # -------------------------
    # 3. Efficiency (SMART)
    # -------------------------
    optimal_steps = 3
    if steps <= optimal_steps:
        efficiency_score = 0.25
    else:
        efficiency_score = max(0, 0.25 - 0.05 * (steps - optimal_steps))

    score += efficiency_score
    logs.append(f"⚡ Efficiency score: {round(efficiency_score,2)}")

    # -------------------------
    # 4. Communication Quality (IMPORTANT)
    # -------------------------
    if any(word in convo_text for word in ["sorry", "apologize", "understand"]):
        score += 0.15
        logs.append("💬 Good customer communication")
    else:
        logs.append("⚠️ Lacks empathy")

    # -------------------------
    # 5. Decision Quality (NEW 🔥)
    # -------------------------
    if "escalate" in convo_text:
        if task_type in ["complex", "edge_case"]:
            score += 0.1
            logs.append("📈 Correct escalation decision")
        else:
            score -= 0.2
            logs.append("⚠️ Unnecessary escalation")

    # -------------------------
    # 6. Flow Realism (VERY IMPORTANT)
    # -------------------------
    if steps < 2:
        score -= 0.2
        logs.append("⚠️ Unrealistic short interaction")

    if steps > max_turns:
        score -= 0.1
        logs.append("⚠️ Too many steps")

    # -------------------------
    # FINAL SCORE
    # -------------------------
    final_score = round(max(0.0, min(score, 1.0)), 2)

    return {
        "final_score": final_score,
        "logs": logs
    }