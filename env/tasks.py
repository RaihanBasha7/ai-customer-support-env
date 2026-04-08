TASKS = [
    {
        "id": "easy_refund",
        "message": "Hi, I received a damaged product and I would like a refund.",
        "type": "refund",
        "max_turns": 4
    },
    {
        "id": "medium_angry",
        "message": "This is the worst service I’ve ever experienced. I’m really frustrated.",
        "type": "angry",
        "max_turns": 5
    },
    {
        "id": "hard_complex",
        "message": "My order is delayed for over a week, and I want both a refund and compensation for the inconvenience.",
        "type": "complex",
        "max_turns": 8
    },
    {
        "id": "edge_case",
        "message": "I see an unauthorized charge on my account and I never placed this order. This might be fraud.",
        "type": "edge_case",
        "max_turns": 6
    }
]