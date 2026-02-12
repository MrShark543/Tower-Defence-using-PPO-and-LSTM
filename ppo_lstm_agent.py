import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
import math
import pygame

import game
from game import GameManager, load_map, interpolate_paths

# Hyperparameters
PPO_CLIP_EPSILON = 0.1  # PPO clip parameter
PPO_EPOCHS = 4          # PPO epochs
GAMMA = 0.99            # Discount factor
GAE_LAMBDA = 0.95       # GAE lambda parameter
CRITIC_DISCOUNT = 0.5   # Value loss coefficient
ENTROPY_BETA = 0.03     # Entropy coefficient for exploration
LR = 1e-4               # Learning rate
BATCH_SIZE = 32         # Batch size for PPO update
HIDDEN_SIZE = 256       # Hidden size for LSTM
SEQUENCE_LENGTH = 8     # Sequence length for LSTM

# Experience data structure
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'action_prob', 'value'])




def test_agent(model_path, num_episodes=5):
    """Test the trained PPO-LSTM agent"""
    # Create game environment
    env = GameManager()
    
    # Get grid dimensions
    global GRID_HEIGHT, GRID_WIDTH
    GRID_HEIGHT = len(env.game_map)
    GRID_WIDTH = len(env.game_map[0])
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent
    action_size = 3 * 10 + 3 * 10 + 1  # Places, upgrades, and do nothing
    agent = PPOLSTMAgent(action_size, device)
    
    # Load trained model
    agent.load(model_path)
    
    # Enable rendering
    game.DISPLAY_GAME = True
    print("Test Mode - RENDERING ENABLED")
    
    # Available maps for testing
    test_maps = ["map1_0", "map1_1", "map1_2", "map1_3", "map1_4"]
    
    # Testing loop
    for episode in range(1, num_episodes + 1):
        # Select random map
        env.current_map = random.choice(test_maps)
        env.game_map, env.all_paths, env.paths = load_map(env.current_map)
        
        # Reset environment and agent
        env.reset_game()
        agent.reset_hidden_states()
        
        # Get initial state
        state = env.get_state()
        
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"Testing Episode {episode}/{num_episodes} on map {env.current_map}")
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Select best action
            action, _, _ = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            

        
        print(f"Episode {episode} finished with score {env.score}, "
              f"lives: {env.player_lives}, wave: {env.wave_number}")


# LSTM + PPO Network
class PPOLSTMNetwork(nn.Module):
    """
    A neural network combining CNN, scalar inputs, and LSTM for PPO-based reinforcement learning.

    This model processes spatial (grid) and scalar game state inputs to predict both the policy (action probabilities)
    and the value function. It uses convolutional layers to extract spatial features from grid input,
    fully connected layers for scalar input, combines both, and passes them through an LSTM to maintain temporal context.

    Attributes:
        conv1, conv2, conv3 (nn.Conv2d): Convolutional layers for grid feature extraction.
        scalar_fc (nn.Linear): Fully connected layer for processing scalar input.
        combine_layer (nn.Linear): Layer for combining CNN and scalar features.
        lstm (nn.LSTM): LSTM layer for sequence processing and temporal learning.
        actor (nn.Linear): Output layer for predicting action probabilities.
        critic (nn.Linear): Output layer for predicting the state value.
        grid_height, grid_width (int): Grid dimensions obtained from the game.
        _initialize_weights(): Initializes weights using orthogonal initialization for stability.

    Methods:
        forward(grid_state, scalar_state, hidden_states=None): 
            Forward pass to produce action probabilities and value estimation with optional LSTM hidden states.

        get_action(grid_state, scalar_state, hidden_states=None): 
            Samples an action from the policy distribution and returns it along with log probability,
            estimated value, and updated hidden states.
    """
    def __init__(self, action_size, scalar_size=6, hidden_size=HIDDEN_SIZE):
        super(PPOLSTMNetwork, self).__init__()
        
        # CNN for processing grid state (4 input channels)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Get grid dimensions from game
        self.grid_height = game.GRID_HEIGHT
        self.grid_width = game.GRID_WIDTH
        
        # CNN output size calculation
        conv_output_size = 32 * self.grid_height * self.grid_width
        
        # Scalar features processing
        self.scalar_fc = nn.Linear(scalar_size, 64)
        
        # Feature combination layer
        self.combine_layer = nn.Linear(conv_output_size + 64, hidden_size)
        
        # LSTM layer 
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Actor (policy) head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights 
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    module.bias.data.zero_()
                    
        # Special initialization for final layers
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)
    
    def forward(self, grid_state, scalar_state, hidden_states=None):
        # Process grid state with CNN
        x1 = F.relu(self.conv1(grid_state))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1)  # Flatten
        
        # Process scalar state
        x2 = F.relu(self.scalar_fc(scalar_state))
        
        # Combine features
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.combine_layer(x))
        
        # Reshape for LSTM if processing sequences
        batch_size = x.size(0)
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)  
        
        # Process with LSTM
        if hidden_states is None:
            self.lstm.flatten_parameters()
            lstm_out, hidden_states = self.lstm(x)
        else:
            self.lstm.flatten_parameters()
            lstm_out, hidden_states = self.lstm(x, hidden_states)
        
        # Get features from LSTM output
        lstm_features = lstm_out.reshape(-1, HIDDEN_SIZE)
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor(lstm_features), dim=-1)
        
        # Critic output (state value)
        state_values = self.critic(lstm_features)
        
        return action_probs, state_values, hidden_states
    
    def get_action(self, grid_state, scalar_state, hidden_states=None):
        with torch.no_grad():
            action_probs, state_value, new_hidden_states = self.forward(grid_state, scalar_state, hidden_states)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
        return action.item(), action_log_prob.item(), state_value.item(), new_hidden_states

# Buffer for storing trajectory data
class TrajectoryBuffer:
    """
    A circular buffer for storing and sampling experience trajectories used in PPO training.

    This buffer holds dictionaries of experience tuples and supports both random sampling for mini-batch training 
    and access to the full buffer contents. It overwrites old data once the capacity is reached, 
    making it efficient for continual learning.

    Attributes:
        capacity (int): Maximum number of experiences the buffer can hold.
        batch_size (int): Number of experiences to sample in each batch.
        buffer (list): Internal list storing experience dictionaries.
        position (int): Index of the next position to insert a new experience.

    Methods:
        push(experience): Adds a new experience to the buffer, overwriting the oldest if full.
        sample_batch(): Returns a randomly sampled batch of experiences of size `batch_size`.
        get_all(): Returns all experiences currently in the buffer.
        __len__(): Returns the current number of stored experiences.
    """
    def __init__(self, capacity, batch_size=BATCH_SIZE):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0
    
    def push(self, experience):
        """Experience dictionary for the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self):
        """Sample a batch of experience dictionaries"""
        batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        return [self.buffer[i] for i in batch_indices]
    
    def get_all(self):
        """Return all experiences in the buffer"""
        return self.buffer
    
    def __len__(self):
        return len(self.buffer)

# PPO Dataset for minibatch training
class PPODataset(Dataset):
    """
    A PyTorch Dataset class for preparing experience data for training a PPO agent with LSTM.

    This class takes a list of experience dictionaries containing states, actions, rewards, and
    related information collected during environment interaction. It processes and converts them 
    into tensors suitable for batched training using a DataLoader.

    Attributes:
        device (torch.device): Device on which tensors are stored (CPU or CUDA).
        grid_states (torch.FloatTensor): Batched grid state tensors of shape [N, C, H, W].
        scalar_states (torch.FloatTensor): Batched scalar state tensors of shape [N, D].
        actions (torch.LongTensor): Actions taken at each timestep.
        rewards (torch.FloatTensor): Rewards received at each timestep.
        dones (torch.FloatTensor): Done flags indicating episode termination.
        old_action_probs (torch.FloatTensor): Log probabilities of actions under the old policy.
        old_values (torch.FloatTensor): Estimated values of states under the old policy.
        returns (torch.FloatTensor): Computed returns using GAE.
        advantages (torch.FloatTensor): Computed advantages using GAE.

    Methods:
        __len__(): Returns the number of samples.
        __getitem__(idx): Returns the tensors corresponding to a single experience at index `idx`.
        _compute_returns_advantages(rewards, values, dones): 
            Computes the discounted returns and normalized advantages using GAE.
    """
    def __init__(self, experiences, device):
        self.device = device
        
        # Separate experiences into components
        grid_states, scalar_states = [], []
        actions, rewards, dones = [], [], []
        action_probs, values = [], []
        
        for exp in experiences:
            grid_states.append(exp['state']['grid_state'])
            scalar_states.append(exp['state']['scalar_state'])
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            dones.append(float(exp['done']))
            action_probs.append(exp['action_prob'])
            values.append(exp['value'])
        
        # Convert to tensors
        self.grid_states = torch.FloatTensor(np.array(grid_states)).to(device)
        self.scalar_states = torch.FloatTensor(np.array(scalar_states)).to(device)
        self.actions = torch.LongTensor(actions).to(device)
        self.rewards = torch.FloatTensor(rewards).to(device)
        self.dones = torch.FloatTensor(dones).to(device)
        self.old_action_probs = torch.FloatTensor(action_probs).to(device)
        self.old_values = torch.FloatTensor(values).to(device)
        
        # Compute returns and advantages
        self.returns, self.advantages = self._compute_returns_advantages(
            rewards, values, dones
        )
    
    def _compute_returns_advantages(self, rewards, values, dones):
        
        returns = []
        advantages = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1 or dones[i]:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + GAMMA * next_value * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return (
            self.grid_states[idx],
            self.scalar_states[idx],
            self.actions[idx],
            self.old_action_probs[idx],
            self.returns[idx],
            self.advantages[idx],
            self.old_values[idx]
        )

# PPO Agent with LSTM
class PPOLSTMAgent:
    """
    A reinforcement learning agent implementing Proximal Policy Optimization (PPO) with LSTM for temporal sequence modeling.

    This agent is designed for environments with spatial and scalar inputs, using a neural network that combines CNNs, 
    fully connected layers, and an LSTM to capture both spatial and temporal patterns. It uses PPO to optimize policy and 
    value networks with clipped surrogate objectives and generalized advantage estimation (GAE).

    Attributes:
        action_size (int): Total number of possible actions.
        device (str or torch.device): Device to run computations on (e.g., "cpu" or "cuda").
        policy (PPOLSTMNetwork): The neural network model combining CNN, LSTM, and linear layers.
        optimizer (torch.optim.Optimizer): Optimizer for updating the network parameters.
        trajectory_buffer (TrajectoryBuffer): Stores experience data for PPO updates.
        hidden_states (tuple): Stores LSTM hidden and cell states.
        batch_size (int): Mini-batch size used during training.
        ppo_epochs (int): Number of epochs per PPO update.
        clip_epsilon (float): Clipping parameter for PPO loss.
        gamma (float): Discount factor for rewards.
        gae_lambda (float): Lambda parameter for GAE.
        critic_discount (float): Weight for the value loss in total loss computation.
        entropy_beta (float): Weight for the entropy bonus to encourage exploration.
        policy_losses (list): History of policy loss values.
        value_losses (list): History of value loss values.
        entropy_losses (list): History of entropy values.
        total_losses (list): History of total loss values.
        avg_returns (list): History of average returns.

    Methods:
        reset_hidden_states(): Resets the LSTM hidden states between episodes.
        select_action(state): Chooses an action based on the current policy and state.
        update_policy(experiences): Performs PPO updates using the collected experiences.
        save(filename): Saves the model weights and optimizer state.
        load(filename): Loads the model weights and optimizer state.
        analyze_architectural_contribution(): Estimates gradient contributions of CNN, LSTM, and scalar parts of the model.
        calculate_convergence_metrics(): Calculates metrics to monitor training stability and convergence.
        track_exploration_exploitation(actions, rewards): Tracks exploration vs. exploitation dynamics based on actions and rewards.
    """
    def __init__(self, action_size, device='cpu', scalar_size=6):
        self.action_size = action_size
        self.device = device
        
        # Add class attributes for hyperparameters
        self.batch_size = BATCH_SIZE  
        self.ppo_epochs = PPO_EPOCHS  
        self.clip_epsilon = PPO_CLIP_EPSILON  
        self.gamma = GAMMA  
        self.gae_lambda = GAE_LAMBDA  
        self.critic_discount = CRITIC_DISCOUNT  
        self.entropy_beta = ENTROPY_BETA 
        self.batch_size = BATCH_SIZE
        self.ppo_epochs = PPO_EPOCHS
        self.scalar_size = scalar_size
        # Policy network
        self.policy = PPOLSTMNetwork(action_size, scalar_size=scalar_size).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)  # Make sure LR is defined, e.g., LR = 3e-4
        
        # Trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(10000, self.batch_size)
        
        # Hidden state management
        self.hidden_states = None
        self.reset_hidden_states()
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.avg_returns = []
    
    def reset_hidden_states(self):
        """Reset the LSTM hidden states between episodes"""
        self.hidden_states = None
    
    def select_action(self, state):
        """Select action based on current policy and state"""
        grid_state = torch.FloatTensor(state['grid_state']).unsqueeze(0).to(self.device)
        scalar_state = torch.FloatTensor(state['scalar_state']).unsqueeze(0).to(self.device)
        
        action, action_log_prob, value, self.hidden_states = self.policy.get_action(
            grid_state, scalar_state, self.hidden_states
        )
        
        # Save policy action probability
        action_prob = action_log_prob
        
        return action, action_prob, value
    
    def update_policy(self, experiences):
        """Update policy using PPO algorithm with experiences stored as dictionaries"""
        # Convert experiences 
        formatted_experiences = []
        
        for exp in experiences:
            # Access dictionary keys instead of attributes
            formatted_experiences.append({
                'state': exp['state'],
                'action': exp['action'],
                'reward': exp['reward'],
                'next_state': exp['next_state'],
                'done': exp['done'],
                'action_prob': exp['action_prob'],
                'value': exp['value']
            })
        
        # Create dataset from experiences
        dataset = PPODataset(formatted_experiences, self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Multiple epochs of PPO updates
        for _ in range(self.ppo_epochs):
            for batch in dataloader:
                grid_states, scalar_states, actions, old_action_probs, returns, advantages, old_values = batch
                
                # Get current predictions
                action_probs, state_values, _ = self.policy(grid_states, scalar_states)
                
                # Extract probability of taken actions
                dist = Categorical(action_probs)
                curr_action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Convert log_probs to probs
                old_probs = torch.exp(old_action_probs)
                curr_probs = torch.exp(curr_action_log_probs)
                
                # Calculate ratio
                ratios = curr_probs / old_probs
                
                # Calculate surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(state_values.squeeze(-1), returns)
                
                # Combined loss
                loss = policy_loss + self.critic_discount * value_loss - self.entropy_beta * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Track losses
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy.item())
                self.total_losses.append(loss.item())
        
        # Record average returns
        self.avg_returns.append(dataset.returns.mean().item())
        
        return {
            'policy_loss': np.mean(self.policy_losses[-len(dataloader):]),
            'value_loss': np.mean(self.value_losses[-len(dataloader):]),
            'entropy': np.mean(self.entropy_losses[-len(dataloader):]),
            'total_loss': np.mean(self.total_losses[-len(dataloader):]),
            'avg_return': self.avg_returns[-1]
        }
    
    def save(self, filename):
        """Save model checkpoint"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def analyze_architectural_contribution(self):
        """Analyze contribution of different architectural components"""
        contributions = {
            'lstm_contribution': 0.0,
            'cnn_contribution': 0.0,
            'scalar_contribution': 0.0
        }
        
        # Calculate gradient magnitudes for different components
        total_grad_norm = 0.0
        lstm_grad_norm = 0.0
        cnn_grad_norm = 0.0
        scalar_grad_norm = 0.0
        
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_norm
                
                if 'lstm' in name:
                    lstm_grad_norm += param_norm
                elif 'conv' in name:
                    cnn_grad_norm += param_norm
                elif 'scalar_fc' in name:
                    scalar_grad_norm += param_norm
        
        # Calculate relative contributions
        if total_grad_norm > 0:
            contributions['lstm_contribution'] = lstm_grad_norm / total_grad_norm
            contributions['cnn_contribution'] = cnn_grad_norm / total_grad_norm
            contributions['scalar_contribution'] = scalar_grad_norm / total_grad_norm
        
        return contributions

    def calculate_convergence_metrics(self):
        """Calculate metrics related to training stability and convergence"""
        # Calculate windowed variance of returns 
        window_size = min(50, len(self.avg_returns))
        if len(self.avg_returns) >= window_size:
            recent_returns = self.avg_returns[-window_size:]
            return_variance = np.var(recent_returns)
        else:
            return_variance = float('inf')
        
        # Calculate policy entropy trend 
        entropy_trend = 0.0
        if len(self.entropy_losses) >= window_size:
            recent_entropy = self.entropy_losses[-window_size:]
            entropy_trend = (np.mean(recent_entropy[:window_size//2]) - 
                             np.mean(recent_entropy[window_size//2:])) / window_size
        
        # Calculate value function accuracy
        value_loss = np.mean(self.value_losses[-window_size:]) if len(self.value_losses) >= window_size else 0
        
        # Calculate policy stability via policy loss
        policy_loss = np.mean(self.policy_losses[-window_size:]) if len(self.policy_losses) >= window_size else 0
        
        return {
            'return_variance': return_variance,
            'entropy_trend': entropy_trend,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'avg_return': np.mean(self.avg_returns[-window_size:]) if len(self.avg_returns) >= window_size else 0
        }

    def track_exploration_exploitation(self, actions, rewards):
        """Track exploration vs exploitation balance"""
        # Count unique actions over time windows
        action_window_size = min(100, len(actions))
        if action_window_size > 0:
            recent_actions = actions[-action_window_size:]
            unique_actions = len(set(recent_actions))
            action_diversity = unique_actions / action_window_size
            
            # Calculate entropy of action distribution
            action_counts = {a: recent_actions.count(a) for a in set(recent_actions)}
            action_probs = [count / action_window_size for count in action_counts.values()]
            action_entropy = -sum(p * math.log(p + 1e-10) for p in action_probs)
        else:
            unique_actions = 0
            action_diversity = 0
            action_entropy = 0
        
        # Measure exploitation via reward improvement
        reward_improvement = 0.0
        if len(rewards) >= 2 * action_window_size and action_window_size > 0:
            prev_window = rewards[-2*action_window_size:-action_window_size]
            curr_window = rewards[-action_window_size:]
            prev_mean = np.mean(prev_window) if prev_window else 0
            if abs(prev_mean) > 1e-10:
                reward_improvement = (np.mean(curr_window) - prev_mean) / abs(prev_mean)
            else:
                reward_improvement = np.mean(curr_window) - prev_mean
        
        # Calculate policy entropy 
        policy_entropy = np.mean(self.entropy_losses[-action_window_size:]) if len(self.entropy_losses) >= action_window_size else 0
        
        return {
            'action_diversity': action_diversity,
            'action_entropy': action_entropy,
            'reward_improvement': reward_improvement,
            'policy_entropy': policy_entropy,
            'unique_action_count': unique_actions,
            'total_actions': action_window_size
        }