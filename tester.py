import torch
import gymnasium as gym
import time
from Actor import Actor  # Assuming your Actor class file

# 1. Constants (Match your environment configuration)
ENV_NAME = "Walker2d-v5"
EPISODES_TO_WATCH = 5

# 2. Initialize the environment with human rendering enabled
env = gym.make(ENV_NAME, render_mode="human",max_episode_steps=100000)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 3. Instantiate the Actor and load the saved weights
actor = Actor(obs_dim, action_dim)
actor.load_state_dict(torch.load("actor_weights.pt"))

# Set the network to evaluation mode (deactivates layers like Dropout or BatchNorm if present)
actor.eval()

# 4. The Evaluation Loop
for episode in range(EPISODES_TO_WATCH):
    observation, info = env.reset()
    done = False
    total_reward = 0
    num_of_steps = 0

    print(f"--- Starting Episode {episode + 1} ---")

    while not done:
        # Convert observation to a PyTorch tensor and add a batch dimension
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        # Disable gradient calculations to save processing power and memory
        with torch.no_grad():
            # Extract only the mean output from your actor
            # If your actor returns a tuple (mean, std), grab index [0]
            action_mean, _ = actor(obs_tensor)

            # Remove the batch dimension and convert back to a NumPy array for Gymnasium
            action = action_mean.squeeze(0).numpy()

        # Step the environment using the deterministic mean action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        num_of_steps += 1

        # Optional: Add a tiny sleep delay if the physics simulation runs too fast for your monitor
        time.sleep(0.01)

    print(f"Episode {episode + 1} finished with Total Reward: {total_reward:.2f} and {num_of_steps} steps")

env.close()