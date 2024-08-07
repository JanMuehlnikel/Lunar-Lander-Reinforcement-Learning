import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import namedtuple, deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.tensor(np.vstack([e.state for e in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([e.action for e in experiences]), dtype=torch.int64)
        rewards = torch.tensor(np.vstack([e.reward for e in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.vstack([e.done for e in experiences]).astype(np.uint8), dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = ReplayBuffer(buffer_size=10000, batch_size=64)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    def act(self, state, eps=0.):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=0.001)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)

def run_dqn(n_episodes=2000, max_trys=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    log_file_path = 'data/logs/training_log.csv'
    with open(log_file_path, 'w') as log_file:
        log_file.write('Episode,Score,Epsilon\n')

        for i_episode in range(1, n_episodes + 1):
            state, _ = env.reset()
            score = 0
            for t in range(max_trys):
                action = agent.act(state, eps)
                next_state, reward, done, _, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)

            log_file.write(f'{i_episode},{score:.2f},{eps:.2f}\n')

            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}", end="")
            if i_episode % 100 == 0:
                print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}")
                torch.save(agent.qnetwork_local.state_dict(), f'data/checkpoints/checkpoint_{i_episode}.pth')

            if np.mean(scores_window) >= 200.0:
                print(f"\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}")
                torch.save(agent.qnetwork_local.state_dict(), 'data/checkpoints/checkpoint_final.pth')
                break

    return scores