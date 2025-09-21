# N-Armed Bandit Problem: Policy Gradient Implementation

## 1. Abstract

This project implements and analyzes the **policy gradient (gradient bandit) algorithm** for the N-armed Gaussian bandit problem. The bandit environment consists of N=20 arms with true reward values drawn from a standard normal distribution and Gaussian noise (σ²=1). We implement the gradient bandit method using softmax policy with baseline for variance reduction. The experimental setup includes T=5000 training steps with learning rate α=0.1. Key metrics evaluated are average reward, percentage of optimal actions, and cumulative regret. Results demonstrate that the gradient bandit algorithm effectively learns to identify and exploit the optimal arm, achieving convergence toward optimal behavior over the training horizon.

## 2. Background

### N-Armed Bandit Problem
The multi-armed bandit problem is defined as follows:
- At each time step t, an agent selects an action A_t ∈ {1, 2, ..., N}
- The environment returns a reward: **R_t ~ N(q*(A_t), σ²)**
- Each arm a has a true reward value q*(a)
- **Objective**: Maximize expected cumulative reward / Minimize cumulative regret

### Gradient Bandit Algorithm
The gradient bandit maintains numerical preferences H_t(a) for each action and uses a softmax policy to select actions based on these preferences. Unlike value-based methods, it directly optimizes the policy parameters using policy gradient techniques.

## 3. Method (Gradient Bandit)

### Policy Definition
The action selection probability follows a softmax distribution:

```
π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b))
```

### Preference Update Rule
The preferences are updated using the policy gradient with baseline:

```
H_{t+1}(a) = {
    H_t(a) + α(R_t - b_t)(1 - π_t(a))    if a = A_t
    H_t(a) - α(R_t - b_t)π_t(a)          if a ≠ A_t
}
```

Where:
- **α**: Learning rate (step size)
- **R_t**: Reward received at time t
- **b_t**: Baseline (average of historical rewards)
- **A_t**: Action taken at time t

### Baseline Calculation
The baseline is computed as the sample mean of all rewards received so far:
```
b_t = (1/t) Σ_{i=1}^t R_i
```

### Implementation Details
- **Initialization**: H_0(a) = 0 for all actions a
- **Action Selection**: Stochastic sampling from softmax distribution (no argmax)
- **Learning Rate**: Constant α throughout training
- **Baseline**: Sample mean for variance reduction

## 4. Experimental Setup (Reproducibility)

### Environment Configuration
- **Number of Arms**: N = 20
- **True Rewards**: q*(a) ~ N(0, 1) for each arm a
- **Reward Noise**: σ² = 1
- **Environment Type**: Stationary (no drift)

### Training Parameters
- **Horizon**: T = 5000 steps
- **Learning Rate**: α = 0.1
- **Runs**: Single run with fixed seed
- **Seeds**: NumPy seed = 40, PyTorch seed = 40

### Evaluation Metrics
1. **Average Reward**: Mean reward per step over training
2. **Percentage of Optimal Actions**: Frequency of selecting the best arm
3. **Cumulative Regret**: Difference between optimal and achieved cumulative reward

### Statistical Analysis
- Moving average with window size = 50 for smoothed visualization
- Final performance evaluated over last 50 steps
- Comparison with random baseline (5% for 20 arms)

### Software Environment
- **Python**: 3.x
- **Dependencies**: NumPy, PyTorch, Matplotlib
- **Operating System**: macOS
- **Hardware**: Standard CPU computation

### Reproducibility Instructions
```bash
# Clone the repository
git clone [repository-url]
cd n_bandit_problem

# Install dependencies
pip install numpy torch matplotlib

# Run the experiment
python main.py
```

### Code Structure
```
main.py
├── generate_bandit()           # Initialize bandit environment
├── get_reward()               # Sample noisy rewards
├── generate_choose()          # Initialize preferences
├── update_choose()            # Policy gradient update
├── train()                    # Main training loop
├── plot_rewards()             # Reward visualization
├── plot_cumulative_rewards()  # Cumulative reward plot
├── plot_optimal_actions()     # Optimal action percentage
└── analyze_performance()      # Performance metrics
```

### Expected Results
The gradient bandit algorithm demonstrates:
- **Learning Convergence**: Gradual improvement in average reward over time
- **Optimal Action Discovery**: Increasing percentage of optimal actions selected
- **Regret Minimization**: Sublinear growth in cumulative regret

The implementation provides comprehensive visualization tools to analyze the agent's learning progress and performance across different metrics.
