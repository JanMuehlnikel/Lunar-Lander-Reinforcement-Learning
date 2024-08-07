import matplotlib.pyplot as plt
import torch
import numpy as np
from src.dqn import QNetwork

def plot_rewards(episodes, scores):
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, scores, marker=None, linestyle='-')
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.title('Total Reward per Episode')
    plt.grid(True)

    plt.savefig('data/images/rewards.png')
    plt.show()

def plot_epsilon(episodes, epsilon):
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, epsilon, marker=None, linestyle='-')
    plt.ylabel('Epsilon')
    plt.xlabel('Episode')
    plt.title('Epsilon Decay')
    plt.grid(True)

    plt.savefig('data/images/epsilon_decay.png')
    plt.show()

def display_model(env, model_path, num_episodes=10):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load the model
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    def select_action(state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
        with torch.no_grad():
            action_values = model(state)
        return np.argmax(action_values.numpy())

    for episode in range(num_episodes):
        state = env.reset()[0] 
        done = False
        total_reward = 0
        while not done:
            env.render() 
            action = select_action(state) 
            next_state, reward, done, _, _ = env.step(action) 
            total_reward += reward 
            state = next_state  
        print(f"Try {episode + 1}: Total Reward: {total_reward}")

    env.close()

