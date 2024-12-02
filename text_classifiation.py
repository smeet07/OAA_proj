import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load and prepare the dataset
dataset = load_dataset("ag_news")
X = dataset["train"]["text"]
y = dataset["train"]["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = torch.FloatTensor(vectorizer.fit_transform(X_train).toarray())
X_test_vec = torch.FloatTensor(vectorizer.transform(X_test).toarray())
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

input_size = 5000
num_classes = 4

class Environment:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        reward = 1.0 if action == self.y[self.current_index] else 0.0
        self.current_index += 1
        done = self.current_index >= len(self.X)
        next_state = self.X[self.current_index] if not done else None
        return next_state, reward, done

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to print classification report
def print_classification_report(y_true, y_pred, title):
    print(f"\nClassification Report for {title}:")
    print(classification_report(y_true, y_pred))

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.policy(x)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

def train_policy_gradient(model, optimizer, env, num_episodes, batch_size=32):
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        for _ in range(batch_size):
            action_probs = model(state.unsqueeze(0))
            action_probs = action_probs * 0.9 + torch.ones_like(action_probs) * 0.1 / action_probs.size(-1)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
            next_state, reward, done = env.step(action)
            reward *= 0.01
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history['loss'].append(loss.item())
        history['reward'].append(sum(rewards))
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}, Total Reward: {sum(rewards):.2f}")
    return history

def train_dqn(model, optimizer, env, num_episodes, batch_size=32, epsilon=0.1, gamma=0.99):
    memory = deque(maxlen=10000)
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        for _ in range(batch_size):
            if random.random() < epsilon:
                action = random.randint(0, num_classes - 1)
            else:
                q_values = model(state.unsqueeze(0))
                action = q_values.argmax().item()
            next_state, reward, done = env.step(action)
            reward *= 0.01
            memory.append((state, action, reward, next_state, done))
            if len(memory) > 32:
                batch = random.sample(memory, 32)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.stack(states)
                next_states = torch.stack([s for s in next_states if s is not None])
                current_q = model(states).gather(1, torch.tensor(actions).unsqueeze(1))
                max_next_q = torch.zeros(32)
                max_next_q[:len(next_states)] = model(next_states).max(1)[0].detach()
                target_q = torch.tensor(rewards) + gamma * max_next_q * (1 - torch.tensor(dones).float())
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def train_actor_critic(model, optimizer, env, num_episodes, batch_size=32):
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        for _ in range(batch_size):
            action_probs, state_value = model(state.unsqueeze(0))
            action_probs = action_probs * 0.9 + torch.ones_like(action_probs) * 0.1 / action_probs.size(-1)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, done = env.step(action)
            reward *= 0.01
            if not done:
                _, next_state_value = model(next_state.unsqueeze(0))
                next_state_value = next_state_value.item()
            else:
                next_state_value = 0.0
            advantage = torch.tensor(reward + 0.99 * next_state_value - state_value.item())
            actor_loss = -torch.log(action_probs[0, action]) * advantage
            critic_loss = torch.tensor(advantage).pow(2)
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        if isinstance(model, ActorCritic):
            action_probs, _ = model(X_test)
            _, predicted = torch.max(action_probs, 1)
        else:
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
    return accuracy_score(y_test, predicted)

# Training
env = Environment(X_train_vec, y_train)
models = {
    "Policy Gradient": PolicyNetwork(input_size, num_classes),
    "DQN": DQN(input_size, num_classes),
    "Actor-Critic": ActorCritic(input_size, num_classes)
}

histories = {}
training_times = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    start_time = time.time()
    if name == "Policy Gradient":
        history = train_policy_gradient(model, optimizer, env, num_episodes=5000)
    elif name == "DQN":
        history = train_dqn(model, optimizer, env, num_episodes=5000)
    else:
        history = train_actor_critic(model, optimizer, env, num_episodes=5000)
    end_time = time.time()
    histories[name] = history
    training_times[name] = end_time - start_time

# Evaluation and Analysis
results = {}
for name, model in models.items():
    accuracy = evaluate_model(model, X_test_vec, y_test)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_times[name]:.2f} seconds")

    # Get predictions
    model.eval()
    with torch.no_grad():
        if isinstance(model, ActorCritic):
            action_probs, _ = model(X_test_vec)
            _, predicted = torch.max(action_probs, 1)
        else:
            outputs = model(X_test_vec)
            _, predicted = torch.max(outputs, 1)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, predicted, name)

    # Print classification report
    print_classification_report(y_test, predicted, name)

# Plot learning curves (reward)
plt.figure(figsize=(12,5))
for name, history in histories.items():
    plt.plot(history['reward'], label=f'{name} Reward')
plt.title('Model Reward over Episodes')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend()
plt.show()

# Plot learning curves (loss)
plt.figure(figsize=(12,5))
for name, history in histories.items():
    plt.plot(history['loss'], label=f'{name} Loss')
plt.title('Model Loss over Episodes')
plt.ylabel('Loss')
plt.xlabel('Episode')
plt.legend()
plt.show()

# Compare model sizes
for name, model in models.items():
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{name} has {num_params} parameters")

# Ensemble prediction
ensemble_pred = torch.zeros(len(X_test_vec), num_classes)
for model in models.values():
    if isinstance(model, ActorCritic):
        action_probs, _ = model(X_test_vec)
    else:
        action_probs = model(X_test_vec)
    ensemble_pred += action_probs.softmax(dim=1)
ensemble_pred /= len(models)
_, ensemble_predicted = torch.max(ensemble_pred, 1)
ensemble_accuracy = accuracy_score(y_test, ensemble_predicted)
print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
plot_confusion_matrix(y_test, ensemble_predicted, "Ensemble Model")
print_classification_report(y_test, ensemble_predicted, "Ensemble Model")