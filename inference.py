def run_inference(env):
    state = env.reset()
    done = False

    while not done:
        action = "reply"  # placeholder
        state, reward, done, _ = env.step(action)

    return state