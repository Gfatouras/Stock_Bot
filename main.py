import gym
import os
from gym import spaces
import numpy as np
import random
import yfinance as yf
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import uuid

# Create directories if they don't exist
for folder in ["charts", "models", "gif"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Generate a random session ID for naming saved files.
session_id = uuid.uuid4().hex[:6]  # 6-character random ID

# Set this flag to True to generate a new model, or False to load an existing model.
GENERATE_NEW_MODEL = True

# Global Adjustable Variables
NUM_EPISODES = 1000
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5
GAMMA = 0.99
TAU = 0.005

# Epsilon parameters (no longer used for action noise, but still kept if you wish to decay OU sigma)
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999

# Data parameters
TICKER = "AMZN"
DATA_PERIOD = "5y"
DATA_INTERVAL = "1d"

BATCH_SIZE = 32
MEMORY_SIZE = 10000

# Global noise scale (for exploration) used as sigma for OU noise.
NOISE_SCALE = 0.25

# ------------------------------
# Ornstein-Uhlenbeck Noise for Exploration
# ------------------------------
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

# ------------------------------
# Prioritized Replay Buffer
# ------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = (state, action, reward, next_state, done)
            self.priorities[idx] = max_priority

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        state, action, reward, next_state, done = map(np.array, zip(*samples))
        action = action.flatten()
        return state, action, reward, next_state, done, indices, weights

    def update_priorities(self, indices, errors, offset=1e-5):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + offset

    def __len__(self):
        return len(self.buffer)

# ------------------------------
# Actor Network (with scaled output)
# ------------------------------
class Actor(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # output one continuous action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = 0.9 * torch.tanh(self.fc3(x))  # scaled output in [-0.9, 0.9]
        return action

# ------------------------------
# Critic Network
# ------------------------------
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + 1, hidden_size)  # state and action
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        return q

# ------------------------------
# DDPG Agent with Prioritized Replay and OU Noise
# ------------------------------
class DDPGAgent:
    def __init__(self, state_size, actor_lr=LR_ACTOR, critic_lr=LR_CRITIC,
                 gamma=GAMMA, tau=TAU, buffer_capacity=MEMORY_SIZE, batch_size=BATCH_SIZE):
        self.state_size = state_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = EPSILON

        self.actor = Actor(state_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.target_actor = Actor(state_size).to(device)
        self.target_critic = Critic(state_size).to(device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        
        # Initialize OU Noise for exploration
        self.ou_noise = OUNoise(size=1, mu=0.0, theta=0.15, sigma=NOISE_SCALE)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            raw_action = self.actor(state_tensor).cpu().data.numpy().flatten()[0]
        self.actor.train()
        # Use OU noise sample
        noise = self.ou_noise.sample()[0]
        effective_action = np.clip(raw_action + noise, -0.9, 0.9)
        return np.array([effective_action], dtype=np.float32)
    
    def reset_noise(self):
        self.ou_noise.reset()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        sample = self.replay_buffer.sample(self.batch_size, beta=0.4)
        if sample is None:
            return
        states, actions, rewards, next_states, dones, indices, weights = sample
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(device).unsqueeze(1)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        td_errors = (current_q - target_value).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        critic_loss = (weights * (current_q - target_value).pow(2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def save_model(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# ------------------------------
# Device Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Trading Environment (Normalized State)
# ------------------------------
class TradingEnvContinuous(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data, initial_balance=None):
        super(TradingEnvContinuous, self).__init__()
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        if initial_balance is None:
            self.initial_balance = float(self.data.iloc[0]['Open'])
        else:
            self.initial_balance = initial_balance
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.reset()
    
    def reset(self):
        print("Data length:", len(self.data))
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.balance
        return self._get_observation()
    
    def _get_observation(self):
        scale = float(self.data.iloc[0]['Open'])
        current_open = float(self.data.iloc[self.current_step]['Open'].item()) / scale
        current_close = float(self.data.iloc[self.current_step]['Close'].item()) / scale
        norm_time = self.current_step / (self.n_steps - 1)
        norm_balance = self.balance / self.initial_balance
        return np.array([current_open, current_close, norm_balance, self.shares_held, norm_time], dtype=np.float32)
    
    def step(self, action):
        trade_fraction = np.clip(float(action[0]), -0.9, 0.9)

        done = False
        terminal_reason = None
        current_open = float(self.data.iloc[self.current_step]['Open'])
        current_close = float(self.data.iloc[self.current_step]['Close'])
        prev_total_value = self.balance + self.shares_held * current_open

        if trade_fraction > 0:
            max_investable = self.balance * 0.99
            amount_to_invest = min(self.balance * trade_fraction, max_investable)
            shares_to_buy = amount_to_invest / current_open
            shares_change = shares_to_buy
        elif trade_fraction < 0:
            effective_fraction = min(-trade_fraction, 0.9)
            shares_to_sell = self.shares_held * effective_fraction
            shares_change = -shares_to_sell
        else:
            shares_change = 0.0

        if shares_change > 0:
            cost = current_open * shares_change
            if self.balance >= cost:
                self.balance -= cost
                self.shares_held += shares_change
            else:
                cost = self.balance * 0.99
                shares_change = cost / current_open
                self.balance -= cost
                self.shares_held += shares_change
        elif shares_change < 0:
            shares_to_sell = min(abs(shares_change), self.shares_held)
            proceeds = current_open * shares_to_sell
            self.balance += proceeds
            self.shares_held -= shares_to_sell

        new_total_value = self.balance + self.shares_held * current_close
        agent_return = new_total_value / prev_total_value - 1
        stock_return = current_close / current_open - 1

        trade_bonus = 0.001 if abs(trade_fraction) > 0.01 else 0.0

        initial_price = float(self.data.iloc[0]['Open'])
        benchmark_value = self.initial_balance * (current_close / initial_price)
        excess_return = (new_total_value - benchmark_value) / benchmark_value
        performance_bonus = 0.01 * excess_return if excess_return > 0 else 0.0

        reward = agent_return - stock_return + trade_bonus + performance_bonus

        self.current_step += 1
        if self.current_step >= self.n_steps:
            done = True
            terminal_reason = "data_exhausted"

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"total_value": new_total_value, "terminal_reason": terminal_reason}
        return obs, reward, done, info

# ------------------------------
# Live Chart Update Function (with Episode Number)
# ------------------------------
def update_live_chart(time_steps, prices, portfolio_values, actions, cash, shares, ax, episode):
    ax.clear()
    prices_arr = np.array(prices)
    n = len(prices_arr)
    cash_arr = np.array(cash) if len(cash) > 0 else np.zeros(n)
    shares_arr = np.array(shares) if len(shares) > 0 else np.zeros(n)
    money_in_market = shares_arr * prices_arr
    total_portfolio = cash_arr + money_in_market

    ax.plot(time_steps, prices, label='Stock Price (Close)', color='blue', alpha=0.5)
    ax.fill_between(time_steps, 0, money_in_market, color='lightgreen', alpha=0.7, label='Money in Market')
    ax.fill_between(time_steps, money_in_market, total_portfolio, color='lightblue', alpha=0.7, label='Cash')
    ax.plot(time_steps, total_portfolio, label='Total Portfolio Value', color='orange')

    buy_indices = [i for i, a in enumerate(actions) if a > 0.01]
    sell_indices = [i for i, a in enumerate(actions) if a < -0.01]
    if buy_indices:
        ax.scatter(np.array(time_steps)[buy_indices],
                   np.array(prices_arr)[buy_indices],
                   marker='^', color='green', s=10, label='Buy')
    if sell_indices:
        ax.scatter(np.array(time_steps)[sell_indices],
                   np.array(prices_arr)[sell_indices],
                   marker='v', color='red', s=10, label='Sell')
    
    initial_cash = float(env.data.iloc[0]['Open'])
    final_portfolio = total_portfolio[-1]
    portfolio_gain = ((final_portfolio - initial_cash) / initial_cash) * 100
    
    ax.text(0.02, 0.95, f"Portfolio Gain: {portfolio_gain:.2f}%", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    # Add current episode information
    ax.text(0.02, 0.88, f"Episode: {episode}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value ($)')
    ax.legend(loc='lower right')
    ax.set_title("Episode Evaluation - Portfolio Composition")
    plt.draw()
    plt.pause(0.01)

# ------------------------------
# Real Stock Data Loader
# ------------------------------
def get_real_stock_data(ticker=TICKER, period=DATA_PERIOD, interval=DATA_INTERVAL):
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError("No data downloaded. Check the ticker, period, or interval.")
    return data[['Open', 'Close']].copy()

# ------------------------------
# Main Training & Evaluation Loop
# ------------------------------
if __name__ == '__main__':
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    
    print(f"Downloading historical data for {TICKER} (period: {DATA_PERIOD}, interval: {DATA_INTERVAL})...")
    stock_data = get_real_stock_data()
    
    env = TradingEnvContinuous(stock_data)
    state_size = env.observation_space.shape[0]
    
    agent = DDPGAgent(state_size, batch_size=BATCH_SIZE, buffer_capacity=MEMORY_SIZE)
    
    model_path = "models/model_" + session_id + ".pth"
    if not GENERATE_NEW_MODEL and os.path.exists(model_path):
        print("Loading existing model from:", model_path)
        agent.load_model(model_path)
    else:
        print("Starting a new model training session.")
    
    total_rewards = []
    extra_title = (f"Episodes: {NUM_EPISODES}, Gamma: {GAMMA}, Tau: {TAU}, Epsilon Decay: {EPSILON_DECAY}, "
                   f"Memory Size: {MEMORY_SIZE}, Batch Size: {BATCH_SIZE}")
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        agent.reset_noise()  # Reset OU noise at episode start
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update()
        
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        print(f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward:.2f}, "
              f"Final Portfolio Value: {info['total_value']:.2f}, Terminal Reason: {info.get('terminal_reason', 'data_exhausted')}, "
              f"Epsilon: {agent.epsilon:.3f}")
        total_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            ax.clear()
            ax.plot(range(1, len(total_rewards) + 1), total_rewards, label='Total Reward per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title(f"{extra_title}\nUp to Episode {episode+1}", fontsize=10)
            ax.legend(loc='lower left')
            chart_filename = f"charts/total_rewards_chart_{session_id}.png"
            plt.savefig(chart_filename)
            print("Saved chart to", chart_filename)
            agent.save_model(model_path)
            print("Saved model to", model_path)
        
        eval_time_steps = []
        eval_prices = []
        eval_portfolio = []
        eval_actions = []
        eval_cash = []
        eval_shares = []

        state_eval = env.reset()
        done_eval = False
        while not done_eval:
            current_step = env.current_step
            current_close = float(env.data.iloc[env.current_step]['Close'])
            current_total_value = env.balance + env.shares_held * current_close
            eval_time_steps.append(current_step)
            eval_prices.append(current_close)
            eval_portfolio.append(current_total_value)
            eval_cash.append(env.balance)
            eval_shares.append(env.shares_held)
            
            action_eval = agent.select_action(state_eval)
            eval_actions.append(action_eval[0])
            state_eval, reward_eval, done_eval, info_eval = env.step(action_eval)

        update_live_chart(eval_time_steps, eval_prices, eval_portfolio, eval_actions, eval_cash, eval_shares, ax, episode+1)
        plt.draw()
        plt.pause(0.01)

        gif_filename = f"gif/episode_{session_id}_{episode+1}.png"
        plt.savefig(gif_filename)
        print("Saved episode evaluation chart to", gif_filename)
        
        plt.pause(0.01)
    
    plt.ioff()
    plt.show()
