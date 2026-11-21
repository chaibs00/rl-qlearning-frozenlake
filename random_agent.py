"""
    A simple Agent that takes random actions in the FrozenLake environment.

"""

import gymnasium as gym
import numpy as np

# Create environment :
env =gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=False,
    render_mode="human",
    reward_schedule=(1, 0, 0)
)

# Initialize the environment
state, info = env.reset()

# Number of episodes
episodes = 20_000

rewards = []

for episode in range(episodes):
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Choose a random action
        action = env.action_space.sample()
        print(f"\n Chosen action: {action}")

        # Execute action
        print(f"Taking action: {action}")
        next_state, reward, terminated, truncated, info = env.step(action)

        # Calculate rewards 
        total_reward += reward

        # Update current state
        state = next_state

    rewards.append(total_reward)

env.close()

# Calculate success rate
success_rate = np.mean(rewards) * 100

print(f"Total episodes: {episodes}")
print(f"Successful episodes: {sum(rewards)}")
print(f"Success rate: {success_rate:.2f}%")

