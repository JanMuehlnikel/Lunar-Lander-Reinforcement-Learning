import torch
import numpy as np
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer
from src.utils import save_model
from src.config import *

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.target_update = TARGET_UPDATE
    
    def select_action(self, state):
        # Implement epsilon-greedy action selection
        pass
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_state_actions = self.policy_net(next_states).argmax(1)
            next_state_values = self.target_net(next_states).gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
            expected_state_action_values = rewards + self.gamma * next_state_values * (1 - dones)
        
        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train(env, agent):
    state = env.reset()
    episode_rewards = []
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, next_state, reward, done)
            
            if len(agent.memory) > BATCH_SIZE:
                agent.optimize_model()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}")
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
    
    save_model(agent.policy_net, 'double_dqn_lunar_lander.pth')