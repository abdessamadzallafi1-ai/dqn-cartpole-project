import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os

if not os.path.exists('results'):
    os.makedirs('results')

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, terminated, truncated):
        done = terminated or truncated
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_results(self, scores, filename='results/training_progress.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(scores)
        plt.title('Performance de l\'agent DQN sur CartPole')
        plt.xlabel('Episode')
        plt.ylabel('Score Total')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Graphique sauvegardé: {filename}")

def main():
    env = gym.make('CartPole-v1', render_mode=None)
    
    # 🔧 FIX: Conversion explicite en int Python pour éviter les erreurs Keras
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    
    agent = DQNAgent(state_size, action_size)
    scores = []
    
    print("Début de l'entraînement DQN...")
    print("=" * 50)
    
    for episode in range(100):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action, reward, next_state, terminated, truncated)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        scores.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode+1}/100 - Score: {total_reward} - Moyenne: {avg_score:.2f} - Epsilon: {agent.epsilon:.3f}")
    
    print("=" * 50)
    print("Entraînement terminé!")
    
    agent.save_results(scores)
    agent.model.save('results/dqn_cartpole_model.h5')
    print("Modèle sauvegardé: results/dqn_cartpole_model.h5")
    
    print(f"\nStatistiques:")
    print(f"Score moyen final: {np.mean(scores[-10:]):.2f}")
    print(f"Meilleur score: {max(scores)}")
    print(f"Score moyen global: {np.mean(scores):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()