"""
    Loading the FrozenLake environment from OpenAI Gym.
"""

import gymnasium as gym


# The enviroment won't be deterministic.
env =gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=True,
    render_mode="human",
    reward_schedule=(1, 0, 0)
)


state, info = env.reset(seed=42)

print("Initial state:", state)
print("State space:", env.observation_space)
print("Action space:", env.action_space)

env.render()

input("Press Enter to close the environment...")

env.close()


