TASKS = [
    {
        # ── Easy ────────────────────────────────────────────────────
        "id":               "easy_refund",
        "difficulty":       "easy",
        "message":          "I want a refund for my product",
        "type":             "refund",
        "max_turns":        4,
        "reply_keywords":   ["refund", "return", "reimburse", "money back", "process"],
        "escalate_allowed": False,
        "description":      "Customer requests a simple product refund.",
    },
    {
        # ── Medium ──────────────────────────────────────────────────
        "id":               "medium_angry",
        "difficulty":       "medium",
        "message":          "Your service is terrible!",
        "type":             "angry",
        "max_turns":        5,
        "reply_keywords":   ["sorry", "apologize", "apologies",
                             "understand", "frustration", "help"],
        "escalate_allowed": False,
        "description":      "Angry customer — agent must de-escalate and apologise.",
    },
    {
        # ── Hard ────────────────────────────────────────────────────
        "id":               "hard_complex",
        "difficulty":       "hard",
        "message":          "My order is delayed and I want a refund and compensation",
        "type":             "complex",
        "max_turns":        8,
        "reply_keywords":   ["delay", "compensation", "refund",
                             "resolve", "apologize", "credit"],
        "escalate_allowed": True,
        "description":      "Multi-step ticket: clarify, resolve, escalate.",
    },
    {
        # ── Edge case ───────────────────────────────────────────────
        "id":               "edge_case",
        "difficulty":       "edge_case",
        "message":          "I never placed this order but I'm being charged",
        "type":             "edge_case",
        "max_turns":        6,
        "reply_keywords":   ["investigate", "charge", "order",
                             "verify", "unauthorized", "fraud"],
        "escalate_allowed": True,
        "description":      "Potential fraud — investigate, verify, escalate.",
    },
]

# Fast lookup by task id
TASK_INDEX = {t["id"]: t for t in TASKS}