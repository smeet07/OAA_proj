import torch
import torch.nn as nn
import torch.optim as optim
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load the IMDB dataset
vocab_size = 10000
max_length = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

class SentimentEnvironment:
    def __init__(self, reviews, labels, max_steps=200):
        self.reviews = reviews
        self.labels = labels
        self.max_steps = max_steps
        self.current_step = 0
        self.current_review = None
        self.current_label = None

    def reset(self):
        self.current_step = 0
        idx = random.randint(0, len(self.reviews) - 1)
        self.current_review = self.reviews[idx]
        self.current_label = self.labels[idx]
        return self.current_review

    def step(self, action):
        reward = 1 if action == self.current_label else 0
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.current_review, reward, done

class LSTMPolicy(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMPolicy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return self.softmax(out)

class LSTMDQN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMDQN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

class LSTMActorCritic(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMActorCritic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        actor_out = self.softmax(self.actor(lstm_out[:, -1, :]))
        critic_out = self.critic(lstm_out[:, -1, :])
        return actor_out, critic_out

def train_policy_gradient(model, optimizer, env, num_episodes):
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            state_tensor = torch.LongTensor(state).unsqueeze(0)
            action_probs = model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
            
            next_state, reward, done = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        returns = torch.tensor(rewards)
        log_probs = torch.stack(log_probs)
        
        loss = -(log_probs * returns).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        history['reward'].append(sum(rewards))
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}, Total Reward: {sum(rewards):.2f}")
    
    return history

def train_dqn(model, optimizer, env, num_episodes, epsilon=0.1, gamma=0.99):
    memory = deque(maxlen=10000)
    history = {'loss': [], 'reward': []}
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        
        while True:
            state_tensor = torch.LongTensor(state).unsqueeze(0)
            
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            
            memory.append((state, action, reward, next_state, done))
            
            if len(memory) > 32:
                batch = random.sample(memory, 32)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.LongTensor(states)
                next_states = torch.LongTensor(next_states)
                
                current_q = model(states).gather(1, torch.tensor(actions).unsqueeze(1))
                max_next_q = model(next_states).max(1)[0].detach()
                
                target_q = torch.tensor(rewards) + gamma * max_next_q * (1 - torch.tensor(dones).float())
                
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        history['loss'].append(np.mean(losses))
        history['reward'].append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {np.mean(losses):.4f}, Total Reward: {total_reward:.2f}")
    
    return history

def train_actor_critic(model, optimizer, env, num_episodes):
    history = {'loss': [], 'reward': []}
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        
        while True:
            state_tensor = torch.LongTensor(state).unsqueeze(0)
            action_probs, state_value = model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
            
            next_state, reward, done = env.step(action)
            
            if not done:
                _, next_state_value = model(torch.LongTensor(next_state).unsqueeze(0))
            else:
                next_state_value = torch.tensor([0.0])
            
            advantage = torch.tensor(reward + 0.99 * next_state_value.item() - state_value.item())
            
            actor_loss = -torch.log(action_probs[0, action]) * advantage
            critic_loss = advantage.pow(2)
            
            loss = actor_loss + critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += reward
            losses.append(loss.item())
            
            if done:
                break
            
            state = next_state
        
        history['loss'].append(np.mean(losses))
        history['reward'].append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {np.mean(losses):.4f}, Total Reward: {total_reward:.2f}")
    
    return history

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.LongTensor(X_test))
        if isinstance(model, LSTMActorCritic):
            outputs, _ = outputs
        _, predicted = torch.max(outputs, 1)
    return predicted, (predicted == torch.tensor(y_test)).float().mean().item()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def print_classification_report(y_true, y_pred, title):
    print(f"\nClassification Report for {title}:")
    print(classification_report(y_true, y_pred))

# Training
env = SentimentEnvironment(X_train, y_train)

embedding_dim = 100
hidden_dim = 128

models = {
    "Policy Gradient": LSTMPolicy(vocab_size, embedding_dim, hidden_dim),
    "DQN": LSTMDQN(vocab_size, embedding_dim, hidden_dim),
    "Actor-Critic": LSTMActorCritic(vocab_size, embedding_dim, hidden_dim)
}

histories = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if name == "Policy Gradient":
        history = train_policy_gradient(model, optimizer, env, num_episodes=1000)
    elif name == "DQN":
        history = train_dqn(model, optimizer, env, num_episodes=1000)
    else:
        history = train_actor_critic(model, optimizer, env, num_episodes=1000)
    
    histories[name] = history

# Evaluation and Analysis
for name, model in models.items():
    predicted, accuracy = evaluate_model(model, X_test, y_test)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    plot_confusion_matrix(y_test, predicted, name)
    print_classification_report(y_test, predicted, name)

# Plot learning curves
plt.figure(figsize=(12,5))
for name, history in histories.items():
    plt.plot(history['reward'], label=f'{name} Reward')
plt.title('Model Rewards over Episodes')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend()
plt.show()