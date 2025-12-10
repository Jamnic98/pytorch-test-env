import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Hyperparameters
# --------------------
ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

episodes = 500
max_steps = 500
gamma = 0.99
lr = 1e-3
hidden_size = 128
value_coef = 0.5
entropy_coef = 0.01

# --------------------
# Actor-Critic Network
# --------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden_size, n_actions)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

# --------------------
# Utilities
# --------------------
def select_action(net, state):
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits, value = net(state_t)
    probs = torch.softmax(logits, dim=1)
    action = torch.multinomial(probs, 1).item()
    log_prob = torch.log(probs[0, action])
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    return action, log_prob, value.squeeze(0), entropy

# --------------------
# Training
# --------------------
env = gym.make(ENV_NAME)
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

net = ActorCritic(obs_size, n_actions).to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=lr)

episode_rewards = []

for ep in range(1, episodes + 1):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []
    entropies = []
    total_reward = 0

    for t in range(max_steps):
        action, log_prob, value, entropy = select_action(net, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)
        rewards.append(reward)
        total_reward += reward
        state = next_state

        if done:
            break

    # Compute returns and advantages
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    values = torch.stack(values)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Loss
    policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    value_loss = nn.MSELoss()(values, returns)
    entropy_loss = torch.stack(entropies).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    optimizer.step()

    episode_rewards.append(total_reward)

    if ep % 10 == 0:
        print(f"Episode {ep}: Reward = {total_reward:.1f}, Avg(20) = {np.mean(episode_rewards[-20:]):.2f}")

env.close()

# --------------------
# Plot rewards
# --------------------
window = 20
moving_avg = [np.mean(episode_rewards[max(0, i - window):i + 1]) for i in range(len(episode_rewards))]

plt.plot(episode_rewards, alpha=0.3, label="Reward per episode")
plt.plot(moving_avg, label=f"Moving average (window={window})")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("CartPole Actor-Critic Training")
plt.legend()
plt.show()
