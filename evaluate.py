
import numpy as np
import matplotlib.pyplot as plt

# 1. Load training history
history = np.load("results/training_history.npz")

rewards = history["rewards"]
successes = history["successes"]

episodes = np.arange(1, len(rewards) + 1)

# 2. Calculate moving average
window_size = 100

moving_success_rate = np.convolve(
    successes,
    np.ones(window_size) / window_size,
    mode="valid"
) * 100

moving_rewards = np.convolve(
    rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)

average_episodes = np.arange(
    window_size, len(rewards) + 1
)

# 3. Plot episode rewards
plt.figure(figsize=(10, 5))

plt.plot(episodes, rewards, alpha=0.3, label="Episode reward")
plt.plot(
    average_episodes,
    moving_rewards,
    linewidth=2,
    label=f"Moving average ({window_size} episodes)"
)

plt.xlabel("Episodes")
plt.ylabel("Total reward")
plt.title("Q-Learning: Training Rewards")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_rewards.png")
plt.show()

# 4. Plot success rate
plt.figure(figsize=(10, 5))

plt.plot(average_episodes, moving_success_rate, linewidth=2)

plt.xlabel("Episodes")
plt.ylabel("Success rate (%)")
plt.title(f"Q-Learning: Success Rate ({window_size}-Episode Moving Average)")
plt.ylim(0, 100)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/success_rate.png")
plt.show()

print("Visualizations saved in results/")
