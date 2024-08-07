"""
IMPORTS
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import glob
from collections import namedtuple, deque

"""
In dieser Ausarbeitung wurde sich maßgeblich am "Playing Atari with Deep Reinforcement Learning" Paper von Volodymyr Mnih et al. orientiert. Im Folgenden ist die 
programmiertechnische Umsetzung des Deep Q-Learning für das Lunar Lander Problem dargestellt und beschrieben.

Link zum Paper: https://arxiv.org/pdf/1312.5602
Link zur Lunar Lander Dokumentation im gymnasium Enviroment: https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""

"""
Das Q-Network ist ein neuronales Netzwerk, das für die Approximation der Q-Funktion verwendet wird. Diese Funktion schätzt den erwarteten zukünftigen Reward 
eines Zustands-Aktions-Paares.
"""
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialisierung des Q-Netzwerkes
        
        Parameter:
        - state_size (int): Anzahl der Dimensionen im State Space.
        - action_size (int): Anzahl der Möglichen Aktionen.
            0: do nothing
            1: fire left orientation engine
            2: fire main engine
            3: fire right orientation engine

        Das Netzwerk besteht aus drei voll verbundenen Schichten (fc). Die Eingabe ist der aktuelle Zustand (State) des Environments, und die Ausgabe sind 
        die Q-Werte für jede mögliche Aktion im aktuellen Zustand.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """
        Forward Pass im Netzwerk
        
        Parameter:
        - state (Tensor): Aktueller State.
        
        Ausgabe:
        - Tensor mit den Werten für die Aktionen von dem eingegebenen State
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

"""
Der Replay Buffer speichert vergangene Erfahrungen (Experiences), die aus Zuständen, Aktionen, Belohnungen, nächsten Zuständen und Episodenenden bestehen. Der Replay Buffer ermöglicht 
das Training mit zufällig gesampelten Experiences. Dies reduziert die Korrelation zwischen aufeinanderfolgenden Erfahrungen und verbessert die Stabilität und 
Effizienz des Lernprozesses.
"""
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Initialisierung des Replay Buffers
        
        Parameter:
        - buffer_size (int): Maximale Anzahl der Experiences die gespeichert werden
        - batch_size (int): Anzahl der Experiences für jeden Trainingsschritt
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """
        Hinzufügen einer Experience zu dem ReplayBuffer

        Parameter:
        - state (array): State vor der Aktion
        - action (int): Durchgeführte Aktion
        - reward (float): Reward nach ausführung der Aktion
        - next_state (array): Nächster State nach Aktion
        - done (bool): Episode beendet?
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Gibt einen Batch von mehreren Experiences aus. Jeder Experience ist eine Liste welche folgend aufgebaut ist:
        (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.tensor(np.vstack([e.state for e in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([e.action for e in experiences]), dtype=torch.int64)
        rewards = torch.tensor(np.vstack([e.reward for e in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.vstack([e.done for e in experiences]).astype(np.uint8), dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

"""
Der Agent interagiert mit der Umgebung, wählt Aktionen aus, sammelt Erfahrungen und aktualisiert das Q-Network basierend auf diesen Erfahrungen. Der Agent enthält 
das Q-Network (lokal und target) und den Replay Buffer. Er verwendet einen Optimierungsalgorithmus (hier Adam), um die Gewichte des Q-Networks zu aktualisieren. Der 
Agent ist damit das zentrale Element im DQN-Algorithmus, das die Strategien zur Entscheidungsfindung und das Lernen von den gesammelten Erfahrungen implementiert.
"""
class Agent:
    def __init__(self, state_size, action_size):
        """
        Initialisierung des Agenten
        
        Parameter:
        - state_size (int): Anzahl der Dimensionen im State Space.
        - action_size (int): Anzahl der Möglichen Aktionen.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = ReplayBuffer(buffer_size=10000, batch_size=64)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Der Agent lernt und Experience wird zum ReplayBuffer hinzugefügt.
        
        Parameter:
        - state (array): State vor der Aktion
        - action (int): Durchgeführte Aktion
        - reward (float): Reward nach ausführung der Aktion
        - next_state (array): Nächster State nach Aktion
        - done (bool): Episode beendet?
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    def act(self, state, eps=0.):
        """
        Auswahl einer Aktion basierend auf dem aktuellen State des Enviroments. Zur Auswahl der Aktion wird die Epsilon-Greedy Policy angewendet, welche zufällige 
        Aktionen (Exploration) und die besten bekannten Aktionen (Exploitation) basierend auf dem aktuellen Q-Network auswählt. Mnih et al. verwenden diese Policy, 
        um das Problem des Entdeckens neuer, möglicherweise besserer Strategien während des Trainings zu lösen. Epsilon (ε) steuert das Verhältnis von Exploration 
        zu Exploitation und wird typischerweise während des Trainings verringert.
        
        Paramete:
        - state (array): Aktueller State
        - eps (float): Explorationsrate der epsilon-greedy policy
        
        Ausgabe:
        - int: Ausgeählte Aktion
        """
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
        """
        Updaten des Q-Networks
        
        Parameter:
        - experiences: Batch von Experiences (states, actions, rewards, next_states, done).
        - gamma (float): Discount Faktor für weitere Rewards.
        """
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
        """
        Das Target Modell wird mit den Gewichten und Parametern des lokalen Modells geupdated
        
        Parameters:
        - local_model (QNetwork): Likales Q-Network.
        - target_model (QNetwork): Target Q-Network.
        - tau (float): Soft update parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)


"""
Der Trainingsprozess besteht aus mehreren wichtigen Schritten, die iterativ wiederholt werden:
1. Erfahrungssammlung (Experience Collection):
    - Der Agent interagiert mit der Umgebung, um Erfahrungen zu sammeln, die im Replay Buffer gespeichert werden.
2. Sampling von Erfahrungen (Experience Sampling):
    - Der Agent sampelt zufällig eine Batch von Erfahrungen aus dem Replay Buffer für das Training.
3. Aktualisierung des Q-Networks (Network Update):
    - Berechnung der Q-Targets unter Verwendung der nächsten Zustände und der Belohnungen. Der Unterschied zwischen den geschätzten Q-Werten und den Q-Zielen wird 
    als Loss verwendet und der Loss wird durch Backpropagation minimiert, und die Gewichte des Q-Networks werden aktualisiert.
4. (Soft) Update des Target Networks:
    - Die Gewichte des Target Q-Networks werden langsam an die Gewichte des lokalen Q-Networks angepasst (soft update). Dies sorgt für eine stabilere Zielwertschätzung.
"""
def run_dqn(n_episodes=2000, max_trys=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Ausführung des Trainings über x Episoden.
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # remove old checkpoint files
    files = glob.glob(os.path.join("data/checkpoints", '*'))
    for f in files:
        os.remove(f)

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
                print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}")
                torch.save(agent.qnetwork_local.state_dict(), 'data/checkpoints/checkpoint_final.pth')
                break

    return scores