class CustomerSupportEnv:
    def __init__(self):
        self.state = None

    def reset(self):
        self.state = {
            "conversation": [],
            "status": "open"
        }
        return self.state

    def step(self, action):
        # action: classify, ask, reply, escalate, close
        reward = 0.0
        done = False

        # TODO: implement logic
        return self.state, reward, done, {}