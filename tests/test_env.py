from env.environment import CustomerSupportEnv

env = CustomerSupportEnv()

def test_deterministic():
    env = CustomerSupportEnv()

    obs1 = env.reset(task_id=0)
    env.step({"action_type": "classify", "content": "refund"})
    result1 = env.step({"action_type": "close", "content": ""})

    env = CustomerSupportEnv()
    obs2 = env.reset(task_id=0)
    env.step({"action_type": "classify", "content": "refund"})
    result2 = env.step({"action_type": "close", "content": ""})

    assert result1[3]["final_score"] == result2[3]["final_score"]

def test_invalid_action():
    env = CustomerSupportEnv()
    env.reset()
    _, reward, _, _ = env.step({"action_type": "invalid", "content": ""})
    assert reward < 0