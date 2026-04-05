from env.environment import CustomerSupportEnv

env = CustomerSupportEnv()

# Reset
obs = env.reset()
print("Initial Observation:")
print(obs)

# Step 1
action = {"action_type": "classify", "content": "refund"}
result = env.step(action)
print("\nStep 1:")
print(result)

# Step 2
action = {"action_type": "reply", "content": "refund processed"}
result = env.step(action)
print("\nStep 2:")
print(result)

# Step 3
action = {"action_type": "close", "content": ""}
result = env.step(action)
print("\nStep 3:")
print(result)