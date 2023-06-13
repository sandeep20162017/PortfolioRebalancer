import torch
import torch.nn as nn
import torch.optim as optim

import json

# Define the PortfolioRebalancingEnvironment class
class PortfolioRebalancingEnvironment:
    def __init__(self, initial_weights, returns):
        self.weights = initial_weights
        self.returns = returns
        self.index = -1
    
    def step(self, action):
        self.index += 1
        if self.index >= len(self.returns):
            done = True
            next_state = self.weights
            reward = 0
        else:
            new_weights = [weight + action[i] for i, weight in enumerate(self.weights)]
            next_state = new_weights
            reward = self.returns[self.index]
            done = self.index == len(self.returns) - 1
        return next_state, reward, done


# Define the PortfolioRebalancingAgent class
class PortfolioRebalancingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.memory = []
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if torch.rand(1) <= self.epsilon:
            return torch.FloatTensor([torch.rand(1) for _ in range(self.action_size)])
        else:
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            _, action = torch.max(q_values, dim=0)
            return action
    
    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = reward + self.gamma * torch.max(self.model(next_state))
            state = torch.FloatTensor(state)
            target_f = self.model(state)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

# Load the training data from the JSON file
def load_training_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    assets = data["assets"]
    returns = data["returns"]
    return assets, returns

# Define the main function
def main():
    # Load the training data
    assets, returns = load_training_data("training_data.json")

    # Set up the portfolio rebalancing environment
    initial_weights = [asset["weight"] for asset in assets]
    env = PortfolioRebalancingEnvironment(initial_weights, returns)

    # Set up the portfolio rebalancing agent
    state_size = len(initial_weights)
    action_size = len(initial_weights)
    agent = PortfolioRebalancingAgent(state_size, action_size)

    # Train the agent
    episodes = 100
    batch_size = 32
    for episode in range(episodes):
        state = env.weights
        for t in range(100):  # Assuming a maximum of 100 time steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    # Save the trained model
    torch.save(agent.model.state_dict(), "trained_model.pt")

if __name__ == "__main__":
    main()
