import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

class NEREnvironment:
    def __init__(self, tokens, tags, max_steps=50):
        self.tokens = tokens
        self.tags = tags
        self.current_tags = ["O"] * len(tokens)
        self.max_steps = max_steps
        self.current_step = 0
        self.tag2idx = {tag: idx for idx, tag in enumerate(set(tags))}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

    def reset(self):
        self.current_tags = ["O"] * len(self.tokens)
        self.current_step = 0
        return self.tokens

    def step(self, action):
        self.current_tags[self.current_step] = self.idx2tag[action]
        self.current_step += 1
        done = self.current_step >= min(self.max_steps, len(self.tokens))
        reward = self.calculate_reward()
        return self.tokens, reward, done

    def calculate_reward(self):
        correct = sum(1 for pred, true in zip(self.current_tags, self.tags) if pred == true)
        return correct / len(self.tags)

class LSTMPolicy(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(LSTMPolicy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out)
        return self.softmax(out)

class LSTMDQN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(LSTMDQN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out)
        return out

class LSTMActorCritic(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(LSTMActorCritic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, num_tags)
        self.critic = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        actor_out = self.softmax(self.actor(lstm_out))
        critic_out = self.critic(lstm_out)
        return actor_out, critic_out

def train_policy_gradient(model, optimizer, env, num_episodes, batch_size=32):
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        state_tensor = torch.LongTensor(tokenizer.encode(state, add_special_tokens=False, padding=True, truncation=True, max_length=512))
        lengths = torch.LongTensor([len(state_tensor)])
        for _ in range(min(batch_size, len(state))):
            action_probs = model(state_tensor.unsqueeze(0), lengths)
            action = torch.multinomial(action_probs[0, _], 1).item()
            log_prob = torch.log(action_probs[0, _, action])
            _, reward, done = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if done:
                break
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

def train_dqn(model, optimizer, env, num_episodes, batch_size=32, epsilon=0.1, gamma=0.99):
    memory = deque(maxlen=10000)
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        state_tensor = torch.LongTensor(tokenizer.encode(state, add_special_tokens=False, padding=True, truncation=True, max_length=512)).unsqueeze(0)
        lengths = torch.LongTensor([len(state_tensor[0])])
        for step in range(min(len(state), env.max_steps)):
            if random.random() < epsilon:
                action = random.randint(0, model.fc.out_features - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor, lengths)
                    action = q_values[0, step].argmax().item()
            next_state, reward, done = env.step(action)
            next_state_tensor = torch.LongTensor(tokenizer.encode(next_state, add_special_tokens=False, padding=True, truncation=True, max_length=512)).unsqueeze(0)
            next_lengths = torch.LongTensor([len(next_state_tensor[0])])
            memory.append((state_tensor, step, action, reward, next_state_tensor, done))
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, steps, actions, rewards, next_states, dones = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                steps = torch.tensor(steps)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                lengths = torch.LongTensor([state.size(1) for state in states.unsqueeze(0)])
                next_lengths = torch.LongTensor([state.size(1) for state in next_states.unsqueeze(0)])
                q_values = model(states, lengths)
                current_q = q_values[torch.arange(q_values.size(0)), steps, :]
                current_q = current_q.gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = model(next_states, next_lengths)
                    next_steps = torch.clamp(steps + 1, max=next_q_values.size(1) - 1)
                    max_next_q = next_q_values[torch.arange(next_q_values.size(0)), next_steps, :].max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            total_reward += reward
            state_tensor = next_state_tensor
            lengths = next_lengths
            if done:
                break
        history['loss'].append(np.mean(losses) if losses else 0)
        history['reward'].append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {np.mean(losses) if losses else 0:.4f}, Total Reward: {total_reward:.2f}")
    return history

def train_actor_critic(model, optimizer, env, num_episodes, batch_size=32):
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        state_tensor = torch.LongTensor(tokenizer.encode(state, add_special_tokens=False, padding=True, truncation=True, max_length=512))
        lengths = torch.LongTensor([len(state_tensor)])
        for step in range(min(batch_size, len(state))):
            action_probs, state_value = model(state_tensor.unsqueeze(0), lengths)
            state_value = state_value.squeeze(-1)
            action = torch.multinomial(action_probs[0, step], 1).item()
            next_state, reward, done = env.step(action)
            if not done:
                next_state_tensor = torch.LongTensor(tokenizer.encode(next_state, add_special_tokens=False, padding=True, truncation=True, max_length=512))
                next_lengths = torch.LongTensor([len(next_state_tensor)])
                _, next_state_value = model(next_state_tensor.unsqueeze(0), next_lengths)
                next_state_value = next_state_value.squeeze(-1)
            else:
                next_state_value = torch.zeros_like(state_value)
            current_state_value = state_value[0, step].item()
            next_state_val = next_state_value[0, step] if step < next_state_value.size(1) else 0.0
            advantage = reward + 0.99 * next_state_val - current_state_value
            advantage = torch.tensor(advantage, requires_grad=False)
            actor_loss = -torch.log(action_probs[0, step, action]) * advantage
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_reward += reward
            losses.append(loss.item())
            if done:
                break
        history['loss'].append(np.mean(losses))
        history['reward'].append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {np.mean(losses):.4f}, Total Reward: {total_reward:.2f}")
    return history

def evaluate_model(model, env):
    model.eval()
    with torch.no_grad():
        state = env.reset()
        done = False
        predicted_tags = []
        state_tensor = torch.LongTensor(tokenizer.encode(state, add_special_tokens=False, padding=True, truncation=True, max_length=512))
        lengths = torch.LongTensor([len(state_tensor)])
        while not done:
            if isinstance(model, LSTMActorCritic):
                action_probs, _ = model(state_tensor.unsqueeze(0), lengths)
            else:
                action_probs = model(state_tensor.unsqueeze(0), lengths)
            action = action_probs[0, len(predicted_tags)].argmax().item()
            predicted_tags.append(action)
            _, _, done = env.step(action)
        return predicted_tags, env.calculate_reward()

# Training
tokens = dataset['train'][0]['tokens']
tags = dataset['train'][0]['ner_tags']
env = NEREnvironment(tokens, tags)
vocab_size = tokenizer.vocab_size
embedding_dim = 256
hidden_dim = 512
num_tags = len(set(tags))

models = {
    "DQN": LSTMDQN(vocab_size, embedding_dim, hidden_dim, num_tags),
    "Actor-Critic": LSTMActorCritic(vocab_size, embedding_dim, hidden_dim, num_tags),
    "Policy Gradient": LSTMPolicy(vocab_size, embedding_dim, hidden_dim, num_tags)
}

histories = {}
training_times = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    if name == "Policy Gradient":
        history = train_policy_gradient(model, optimizer, env, num_episodes=1000)
    elif name == "DQN":
        history = train_dqn(model, optimizer, env, num_episodes=1000)
    else:
        history = train_actor_critic(model, optimizer, env, num_episodes=1000)
    end_time = time.time()
    histories[name] = history
    training_times[name] = end_time - start_time

# Evaluation and analysis
for name, model in models.items():
    predicted_tags, accuracy = evaluate_model(model, env)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_times[name]:.2f} seconds")
    print(classification_report(tags, [env.idx2tag[tag] for tag in predicted_tags]))
    plot_confusion_matrix(tags, [env.idx2tag[tag] for tag in predicted_tags], f"{name} Confusion Matrix")
plt.figure(figsize=(12,5))

for name, history in histories.items():
    plt.plot(history['reward'], label=f'{name} Reward')
plt.title('Model Rewards over Episodes')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
for name, history in histories.items():
    plt.plot(history['loss'], label=f'{name} Loss')
plt.title('Model Loss over Episodes')
plt.ylabel('Loss')
plt.xlabel('Episode')
plt.legend()
plt.show()  

print(f"\nOriginal Tags: {tags}")
for name, model in models.items():
    predicted_tags, _ = evaluate_model(model, env)
    print(f"\n{name} Predicted Tags: {[env.idx2tag[tag] for tag in predicted_tags]}")
for name, model in models.items():
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{name} has {num_params} parameters")

