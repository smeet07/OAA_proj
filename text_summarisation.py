import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import BartTokenizer
from rouge_score import rouge_scorer
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

class SummarizationEnvironment:
    def __init__(self, article, reference_summary, max_steps=50):
        self.article = article
        self.reference_summary = reference_summary
        self.current_summary = ""
        self.max_steps = max_steps
        self.current_step = 0
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def reset(self):
        self.current_summary = ""
        self.current_step = 0
        return self.article

    def step(self, action):
        self.current_summary += action + " "
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = self.calculate_reward()
        return self.current_summary, reward, done

    def calculate_reward(self):
        scores = self.scorer.score(self.reference_summary, self.current_summary)
        return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3

class LSTMPolicy(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMPolicy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out[:, -1, :])
        return self.softmax(out)

class LSTMDQN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMDQN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMActorCritic(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMActorCritic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, vocab_size)
        self.critic = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        actor_out = self.softmax(self.actor(lstm_out[:, -1, :]))
        critic_out = self.critic(lstm_out[:, -1, :])
        return actor_out, critic_out

def train_policy_gradient(model, optimizer, env, num_episodes, batch_size=32):
    history = {'loss': [], 'reward': []}
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        for _ in range(batch_size):
            state_tensor = torch.LongTensor(tokenizer.encode(state, return_tensors="pt", max_length=512, truncation=True)[0])
            lengths = torch.LongTensor([len(state_tensor)])
            action_probs = model(state_tensor.unsqueeze(0), lengths)
            
            # Ensure valid probabilities
            epsilon = 1e-8
            action_probs = action_probs + epsilon
            action_probs = action_probs / action_probs.sum()
            action_probs = torch.clamp(action_probs, 0, 1)
            
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
            
            next_state, reward, done = env.step(tokenizer.decode([action]))
            
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

def train_dqn(model, optimizer, env, num_episodes, epsilon=0.1, gamma=0.99):
    memory = deque(maxlen=10000)
    history = {'loss': [], 'reward': []}
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        
        while True:
            state_tensor = torch.LongTensor(tokenizer.encode(state, return_tensors="pt", max_length=512, truncation=True)[0])
            lengths = torch.LongTensor([len(state_tensor)])
            
            if random.random() < epsilon:
                action = random.randint(0, tokenizer.vocab_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor.unsqueeze(0), lengths)
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(tokenizer.decode([action]))
            
            next_state_tensor = torch.LongTensor(tokenizer.encode(next_state, return_tensors="pt", max_length=512, truncation=True)[0])
            next_lengths = torch.LongTensor([len(next_state_tensor)])
            
            memory.append((state_tensor, lengths, action, reward, next_state_tensor, next_lengths, done))
            
            if len(memory) > 32:
                batch = random.sample(memory, 32)
                states, state_lengths, actions, rewards, next_states, next_state_lengths, dones = zip(*batch)
                
                states = pad_sequence(states, batch_first=True)
                state_lengths = torch.cat([l for l in state_lengths])
                next_states = pad_sequence(next_states, batch_first=True)
                next_state_lengths = torch.cat([l for l in next_state_lengths])
                
                current_q = model(states, state_lengths).gather(1, torch.tensor(actions).unsqueeze(1))
                
                with torch.no_grad():
                    max_next_q = model(next_states, next_state_lengths).max(1)[0]
                
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

def train_actor_critic(model, optimizer, env, num_episodes):
    history = {'loss': [], 'reward': []}
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        losses = []
        
        while True:
            state_tensor = torch.LongTensor(tokenizer.encode(state, return_tensors="pt", max_length=512, truncation=True)[0])
            lengths = torch.LongTensor([len(state_tensor)])
            action_probs, state_value = model(state_tensor.unsqueeze(0), lengths)
            
            # Ensure valid probabilities
            epsilon = 1e-8
            action_probs = action_probs + epsilon
            action_probs = action_probs / action_probs.sum()
            action_probs = torch.clamp(action_probs, 0, 1)
            
            action = torch.multinomial(action_probs, 1).item()
            
            next_state, reward, done = env.step(tokenizer.decode([action]))
            
            if not done:
                next_state_tensor = torch.LongTensor(tokenizer.encode(next_state, return_tensors="pt", max_length=512, truncation=True)[0])
                next_lengths = torch.LongTensor([len(next_state_tensor)])
                _, next_state_value = model(next_state_tensor.unsqueeze(0), next_lengths)
            else:
                next_state_value = torch.tensor([0.0])
            
            advantage = torch.tensor(reward + 0.99 * next_state_value.item() - state_value.item())
            
            actor_loss = -torch.log(action_probs[0, action]) * advantage
            critic_loss = torch.square(advantage)
            
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

def evaluate_model(model, env):
    model.eval()
    with torch.no_grad():
        state = env.reset()
        done = False
        generated_summary = ""
        
        while not done:
            state_tensor = torch.LongTensor(tokenizer.encode(state, return_tensors="pt", max_length=512, truncation=True)[0])
            lengths = torch.LongTensor([len(state_tensor)])
            if isinstance(model, LSTMActorCritic):
                action_probs, _ = model(state_tensor.unsqueeze(0), lengths)
            else:
                action_probs = model(state_tensor.unsqueeze(0), lengths)
            action = torch.multinomial(action_probs, 1).item()
            
            next_state, reward, done = env.step(tokenizer.decode([action]))
            generated_summary += tokenizer.decode([action]) + " "
            state = next_state
        
    return generated_summary.strip(), env.calculate_reward()

# Training
article = dataset['train'][0]['article']
reference_summary = dataset['train'][0]['highlights']
env = SummarizationEnvironment(article, reference_summary)

vocab_size = tokenizer.vocab_size
embedding_dim = 256
hidden_dim = 512

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
        history = train_policy_gradient(model, optimizer, env, num_episodes=50)
    elif name == "DQN":
        history = train_dqn(model, optimizer, env, num_episodes=50)
    else:
        history = train_actor_critic(model, optimizer, env, num_episodes=50)
    
    histories[name] = history

# Evaluation
for name, model in models.items():
    generated_summary, rouge_score = evaluate_model(model, env)
    print(f"\n{name} Results:")
    print(f"Generated Summary: {generated_summary}")
    print(f"ROUGE Score: {rouge_score:.4f}")

# Plot learning curves
plt.figure(figsize=(12,5))
for name, history in histories.items():
    plt.plot(history['reward'], label=f'{name} Reward')
plt.title('Model Rewards over Episodes')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend()
plt.show()

# Print reference summary
print(f"\nReference Summary: {reference_summary}")