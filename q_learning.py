
"""
Implementing Q-Learning: Teaching the agent to learn.
"""

import gymnasium as gym
import numpy as np
import os

# 1. Create results directory
os.makedirs("results", exist_ok=True)

# 2. Create environment
env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    is_slippery=True,
    reward_schedule=(1, 0, 0)
)

# 3. Hyperparameters
episodes = 20_000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01


# 4. Initialize Q-table (16 states x 4 actions)
q_table = np.zeros(
    (env.observation_space.n, env.action_space.n)
)

# Training history
episode_rewards = []
episode_successes = []

# 5. Training loop
for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Exploration vs exploitation
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Best estimated future value
        next_max = np.max(q_table[next_state])

        # Q-learning update
        q_table[state, action] += learning_rate * (
            reward + discount_factor * next_max
            - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    # Update exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Record episode results
    episode_rewards.append(total_reward)
    episode_successes.append(1 if total_reward > 0 else 0)

    if (episode + 1) % 1000 == 0:
        recent_success = np.mean(episode_successes[-1000:]) * 100
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Success rate: {recent_success:.2f}% | "
            f"Epsilon: {epsilon:.3f}"
        )

# 6. Save Q-table
np.save("results/q_table.npy", q_table)

# 7. Save training history
np.savez(
    "results/training_history.npz",
    rewards=episode_rewards,
    successes=episode_successes
)

env.close()

# 8. Display results
print("\n=========== Training Results ===========")
print("Q-table:\n", q_table)

print(
    "\nOverall training success rate:",
    np.mean(episode_successes) * 100, "%"
)

print("Final epsilon:", epsilon)

print("\nFiles saved in results/")
