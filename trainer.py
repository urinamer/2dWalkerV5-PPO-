import torch
import torch.nn as nn
import gymnasium as gym
from matplotlib.figure import figaspect
from torch.distributions import Normal
from Actor import Actor
from Critic import Critic
from RollOutBuffer import RollOutBuffer
import matplotlib.pyplot as plt


env = gym.make("Walker2d-v5")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# models & single optimizer setup
actor = Actor(obs_dim, action_dim)
critic = Critic(obs_dim)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(),lr = 3e-4)

critic_loss_fn = nn.HuberLoss()
buffer = RollOutBuffer(size=2048, obs_dim=obs_dim, action_dim=action_dim)
#learning rate decay
scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=3000)


#data for plotting
num_of_ppo_elements = 0
num_of_clipped = 0
losses = [0.0]
rewards = [0]
clipped_fractions = [0.0]


def ppo_loss(advantage, old_log_prob, new_log_prob, clip_epsilon=0.2):
    global num_of_clipped
    global num_of_ppo_elements

    ratio = torch.exp(new_log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage

    with torch.no_grad():
        num_of_clipped += ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).sum().item()# gets number of times clipped
        num_of_ppo_elements += ratio.numel()
    return -torch.min(surr1, surr2).mean()

def plot_training_data(rewards,losses,clip_fractions):
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (18,5))

    ax1.plot(rewards,color = "green",alpha = 0.6)
    ax1.set_title("Total Reward per Episode")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward Score")
    ax1.grid(True)  # Adds a clean background grid


    ax2.plot(losses,color = "red",alpha =0.6)
    ax2.set_title("Average Actor loss")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Average Loss")
    ax2.grid(True)

    ax3.plot(clip_fractions, color="orange", alpha=0.6)
    ax3.set_title("Clip fraction")
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Percentage Clipped")
    ax3.grid(True)


    plt.tight_layout()

    plt.show()


# Core execution loop
current_obs, info = env.reset()

for episode in range(3000):
    # Phase 1: Sequential Experience Collection
    total_rewards = 0
    sum_actor_loss = 0
    num_of_steps = 1
    num_of_ppo_elements = 0
    num_of_clipped = 0
    for _ in range(2048):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(current_obs)
            mean, std = actor(obs_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)  # Summing joints
            value = critic(obs_tensor)

        next_obs, reward, terminated, truncated, info = env.step(action.numpy())
        done = terminated or truncated

        buffer.store(current_obs, action.numpy(), log_prob.item(), reward, done, value.item())

        if done:
            current_obs, info = env.reset()
        else:
            current_obs = next_obs

        total_rewards += reward


    # Phase 2: Compute GAE before splitting data
    with torch.no_grad():
        last_value = critic(torch.FloatTensor(current_obs)).item()
    buffer.calculateAdvantagesAndReturns(last_value, done)

    #Advantage normalization
    buffer.normalize_advantages()


    # Phase 3: Shuffled Mini-Batch Optimization Updates
    batch_size = 100
    for epoch in range(5):
        # Pythonic way to exhaust a generator loop completely
        for obs, action, old_log_prob, value, advantage, target in buffer.get_batches(batch_size):
            # Recalculate predictions with current network states
            new_mean, new_std = actor(obs)
            new_dist = Normal(new_mean, new_std)
            new_log_prob = new_dist.log_prob(action).sum(axis=-1)

            new_value = critic(obs).squeeze()

            # Losses
            actor_loss = ppo_loss(advantage, old_log_prob, new_log_prob)
            critic_loss = critic_loss_fn(new_value, target)  # Or value-clipping loss

            total_loss = actor_loss + critic_loss

            sum_actor_loss += actor_loss.sum().item()
            num_of_steps += batch_size


            # Optimization Step
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            total_loss.backward()
            critic_optimizer.step()
            actor_optimizer.step()

    rewards.append(total_rewards)
    losses.append(sum_actor_loss / num_of_steps)
    clipped_fractions.append(num_of_clipped/num_of_ppo_elements)
    buffer.clear()

torch.save(actor.state_dict(),"actor_weights.pt")
torch.save(critic.state_dict(),"critic_weights.pt")
plot_training_data(rewards,losses,clipped_fractions)