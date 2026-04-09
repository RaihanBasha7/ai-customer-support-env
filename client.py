import requests

BASE_URL = "http://localhost:7860"

def reset():
    response = requests.post(f"{BASE_URL}/reset")
    return response.json()

def step(action):
    response = requests.post(f"{BASE_URL}/step", json=action)
    return response.json()


if __name__ == "__main__":
    state = reset()
    print("Initial State:", state)

    done = False

    while not done:
        action = {
            "action_type": "classify",
            "content": "refund"
        }

        result = step(action)
        print("Step Result:", result)

        done = result.get("done", False)