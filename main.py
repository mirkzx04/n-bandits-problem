# Import necessary libraries
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_bandit(n_arms):
    """
    Generate a bandit problem with n_arms.
    Each arm has a true reward value drawn from a normal distribution.
    
    Args:
        n_arms (int): Number of arms in the bandit problem
        
    Returns:
        np.array: True reward values (q*) for each arm
    """
    np.random.seed(40)
    # Generate true reward values from standard normal distribution
    q_star = np.random.normal(size=n_arms)
    return q_star

def get_reward(q_star, a):
    """
    Get a noisy reward from pulling arm 'a'.
    The reward is sampled from a normal distribution centered at the true value q_star[a].
    
    Args:
        q_star (np.array): True reward values for each arm
        a (int): Index of the arm to pull
        
    Returns:
        float: Noisy reward sampled from N(q_star[a], 1.0)
    """
    return np.random.normal(loc=q_star[a], scale=1.0)

def generate_choose(n_arms):
    """
    Initialize the policy parameters (preferences) for each arm.
    These parameters will be used in a softmax policy to select actions.
    
    Args:
        n_arms (int): Number of arms in the bandit problem
        
    Returns:
        torch.Tensor: Initial preferences for each arm (all zeros)
    """
    th.random.manual_seed(40)
    return th.zeros(n_arms, dtype=th.float32)

def update_choose(choose, rewards, n_arms, q_star, alpha=0.1):
    """
    Update the policy parameters using the policy gradient method.
    This implements the REINFORCE algorithm for the bandit problem.
    
    Args:
        choose (torch.Tensor): Current policy parameters (preferences)
        rewards (torch.Tensor): History of rewards received so far
        n_arms (int): Number of arms in the bandit problem
        q_star (np.array): True reward values for each arm
        alpha (float): Learning rate for policy updates
        
    Returns:
        tuple: (reward, action_idx) - reward received and action taken
    """
    # Convert preferences to action probabilities using softmax
    prob = th.softmax(choose, dim=0)
    # Calculate baseline as mean of historical rewards (for variance reduction)
    baseline = rewards.mean()

    # Sample an action according to the current policy (softmax probabilities)
    action_idx = np.random.choice(n_arms, p=prob.detach().numpy())
    action_reward = get_reward(q_star, action_idx)

    # Update preferences using policy gradient rule
    for a in range(n_arms):
        if a == action_idx:
            choose[a] += alpha * (action_reward - baseline) * (1 - prob[a])
        else:
            choose[a] -= alpha * (action_reward - baseline) * prob[a]

    return action_reward, action_idx

def train(n_arms=10, n_steps=2000, alpha=0.1):
    """
    Train a policy gradient agent on the n-armed bandit problem.
    
    Args:
        n_arms (int): Number of arms in the bandit problem
        n_steps (int): Number of training steps to run
        alpha (float): Learning rate for policy updates
        
    Returns:
        tuple: (rewards, actions, q_star) where:
            - rewards: Array of rewards received at each step
            - actions: Array of actions taken at each step
            - q_star: True reward values for each arm
    """
    # Generate the bandit problem (true reward values)
    q_star = generate_bandit(n_arms)
    # Initialize policy parameters (preferences)
    choose = generate_choose(n_arms)
    
    rewards = []
    actions = []  # Track actions taken

    # Training loop
    for step in range(n_steps):
        reward, action = update_choose(choose, th.tensor(rewards) if rewards else th.tensor([0.0]), n_arms, q_star, alpha)
        rewards.append(reward)
        actions.append(action)

    return np.array(rewards), np.array(actions), q_star

def plot_rewards(rewards, q_star, window_size=50):
    """
    Plot the reward progression over training steps with moving average.
    
    Args:
        rewards (np.array): Array of rewards received at each step
        q_star (np.array): True reward values for each arm
        window_size (int): Size of the moving average window for smoothing
    """
    steps = np.arange(len(rewards))
    
    # Calculate moving average for smoother visualization
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    moving_avg_steps = steps[window_size-1:]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards (semi-transparent)
    plt.plot(steps, rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
    
    # Plot moving average
    plt.plot(moving_avg_steps, moving_avg, color='darkblue', linewidth=2, 
             label=f'Moving Average (window={window_size})')
    
    # Add horizontal line for optimal reward (max of true values)
    plt.axhline(y=np.max(q_star), 
                color='red', linestyle='--', alpha=0.7, 
                label='Optimal Reward (best arm)')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('Policy Gradient Agent Performance on N-Armed Bandit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cumulative_rewards(rewards):
    """
    Plot the cumulative rewards over training steps.
    
    Args:
        rewards (np.array): Array of rewards received at each step
    """
    cumulative_rewards = np.cumsum(rewards)
    steps = np.arange(len(rewards))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cumulative_rewards, color='green', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_performance(rewards, actions, q_star):
    """
    Analyze and display performance metrics of the training.
    
    Args:
        rewards (np.array): Array of rewards received at each step
        actions (np.array): Array of actions taken at each step
        q_star (np.array): True reward values for each arm
    """
    optimal_reward = np.max(q_star)
    optimal_action = np.argmax(q_star)
    optimal_choices = (actions == optimal_action).astype(int)
    
    print("=== Performance Analysis ===")
    print(f"Total steps: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Optimal reward: {optimal_reward:.3f}")
    print(f"Regret (optimal - average): {optimal_reward - np.mean(rewards):.3f}")
    print(f"Final 100 steps average: {np.mean(rewards[-100:]):.3f}")
    print(f"Best arm value: {optimal_reward:.3f} (arm {optimal_action})")
    print(f"Worst arm value: {np.min(q_star):.3f} (arm {np.argmin(q_star)})")
    print(f"% Optimal actions: {np.mean(optimal_choices) * 100:.1f}%")

def plot_optimal_actions(actions, q_star, window_size=50):
    """
    Plot the percentage of optimal actions taken over time.
    
    Args:
        actions (np.array): Array of actions taken at each step
        q_star (np.array): True reward values for each arm
        window_size (int): Size of the moving average window for smoothing
    """
    # Find the optimal action (arm with highest true reward)
    optimal_action = np.argmax(q_star)
    
    # Create binary array: 1 if optimal action was taken, 0 otherwise
    optimal_choices = (actions == optimal_action).astype(int)
    
    # Calculate moving average of optimal action percentage
    moving_avg = np.convolve(optimal_choices, np.ones(window_size)/window_size, mode='valid') * 100
    steps = np.arange(len(actions))
    moving_avg_steps = steps[window_size-1:]
    
    # Calculate cumulative percentage of optimal actions
    cumulative_optimal = np.cumsum(optimal_choices) / (steps + 1) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Moving average of optimal actions
    plt.subplot(2, 1, 1)
    plt.plot(moving_avg_steps, moving_avg, color='purple', linewidth=2, 
             label=f'Moving Average (window={window_size})')
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                label='Perfect Performance (100%)')
    plt.axhline(y=10, color='gray', linestyle='--', alpha=0.5, 
                label='Random Performance (10%)')
    plt.xlabel('Training Steps')
    plt.ylabel('% Optimal Actions')
    plt.title('Percentage of Optimal Actions Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # Subplot 2: Cumulative percentage
    plt.subplot(2, 1, 2)
    plt.plot(steps, cumulative_optimal, color='orange', linewidth=2, 
             label='Cumulative % Optimal')
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                label='Perfect Performance (100%)')
    plt.axhline(y=10, color='gray', linestyle='--', alpha=0.5, 
                label='Random Performance (10%)')
    plt.xlabel('Training Steps')
    plt.ylabel('Cumulative % Optimal Actions')
    plt.title('Cumulative Percentage of Optimal Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    final_window_optimal = np.mean(optimal_choices[-window_size:]) * 100
    overall_optimal = np.mean(optimal_choices) * 100
    
    print(f"\n=== Optimal Action Analysis ===")
    print(f"Optimal arm: {optimal_action} (value: {q_star[optimal_action]:.3f})")
    print(f"Overall % optimal actions: {overall_optimal:.1f}%")
    print(f"Final {window_size} steps % optimal: {final_window_optimal:.1f}%")
    print(f"Random baseline: {100/len(q_star):.1f}%")

# Example usage and testing
if __name__ == "__main__":
    print("Training Policy Gradient Agent on 10-Armed Bandit...")
    
    # Train the agent
    rewards, actions, q_star = train(n_arms=20, n_steps=5000, alpha=0.1)
    
    # Analyze performance
    analyze_performance(rewards, actions, q_star)
    
    # Plot results
    plot_rewards(rewards, q_star, window_size=50)
    plot_cumulative_rewards(rewards)
    plot_optimal_actions(actions, q_star, window_size=50)
    plot_optimal_actions(rewards, window_size=50)