import torch
import gymnasium as gym
import time
from Actor import Actor

ENV_NAME = "Walker2d-v5"
EPISODES_TO_WATCH = 5

env = gym.make(ENV_NAME, render_mode="human",max_episode_steps=100000)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = Actor(obs_dim, action_dim)
actor.load_state_dict(torch.load("actor_weights.pt"))

# set the network to evaluation mode (deactivates layers like Dropout or BatchNorm if present)
actor.eval()

for episode in range(EPISODES_TO_WATCH):
    observation, info = env.reset()
    done = False
    total_reward = 0
    num_of_steps = 0

    print(f"--- Starting Episode {episode + 1} ---")

    while not done:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        with torch.no_grad():
            action_mean, _ = actor(obs_tensor)
            action = action_mean.squeeze(0).numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        num_of_steps += 1

        time.sleep(0.01)

    print(f"Episode {episode + 1} finished with Total Reward: {total_reward:.2f} and {num_of_steps} steps")

env.close()