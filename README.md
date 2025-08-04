# Deep Reinforcement Learning with DQN Variants

This project implements and analyses **Deep Q-Network (DQN)** algorithms and their variants for reinforcement learning tasks. The implementation focusses on the **CartPole** and **Lunar Lander** environments from OpenAI Gymnasium, providing comprehensive hyperparameter analysis and performance evaluation.

## 📋 Project Overview

### 🎯 Key Achievements
- **Multi-Environment DQN Implementation**: Successfully trained agents for CartPole-v0, CartPole-v1, and LunarLander-v3
- **Algorithm Variants**: Implemented vanilla DQN, Duelling DQN, and Double DQN architectures
- **Comprehensive Hyperparameter Analysis**: Systematic exploration of learning rates, epsilon decay, target network synchronisation, and network architectures
- **Performance Optimisation**: Achieved target rewards across all environments with optimised hyperparameters

---

## 🛠️ Technical Implementation

### Core Architecture
```
├── dqn_gym.py          # Main training loop and environment interface
├── dqn_common.py       # Neural network architectures (Single/Two-layer/Duelling DQN)
├── config/dqn.yaml     # Hyperparameter configurations for each environment
├── lib/experience_buffer.py  # Experience replay buffer implementation
└── saved_models/       # Trained model checkpoints
```

### Neural Network Architectures
1. **Single Hidden Layer DQN**: Fast convergence for simple environments
2. **Two Hidden Layer DQN**: Enhanced representation learning
3. **Duelling DQN**: Separate value and advantage estimation networks

### Key Features Implemented
- **Experience Replay Buffer**: Efficient memory management for training stability
- **Target Network**: Stabilised Q-learning with periodic/soft updates
- **Epsilon-Greedy Exploration**: Configurable exploration-exploitation balance
- **Live Training Visualisation**: Real-time R100 performance plotting
- **Multi-Device Support**: GPU/MPS/CPU training compatibility

---

## 🚀 Quick Start

### Requirements
Install dependencies using conda:
```bash
conda env create -f environment.yml
conda activate dqn-env
```

For GPU support, follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### Running Experiments

**Basic CartPole training:**
```bash
python dqn_gym.py -e CartPole-v1 -n single-hidden
```

**Duelling DQN on Lunar Lander:**
```bash
python dqn_gym.py -e LunarLander-v3 -n duelling-dqn
```

**With live plotting:**
```bash
python dqn_gym.py -e CartPole-v1 -n duelling-dqn -v
```

### Command Line Options
- `-e, --env`: Environment name (CartPole-v0, CartPole-v1, LunarLander-v3)
- `-n, --network`: Network architecture (single-hidden, two-hidden, duelling-dqn)
- `-v, --live_plot`: Enable live plotting during training
- `-s, --seed`: Manual seed for reproducibility

---

## 📊 Experimental Results

### Environment Performance
| Environment | Target Reward | Achieved R100 | Training Episodes | Best Architecture |
|-------------|---------------|---------------|-------------------|-------------------|
| CartPole-v0 | 195.0 | 196.32 | ~500 | Single Hidden Layer |
| CartPole-v1 | 475.0 | 475.83 | ~1,200 | Duelling DQN |
| LunarLander-v3 | 200.0 | 202.98 | ~2,500 | Duelling DQN |

### Hyperparameter Optimisation Results
- **Learning Rate Analysis**: Optimal rates varied by environment (5e-4 to 1e-3)
- **Epsilon Decay**: Longer decay schedules improved final performance
- **Target Network Updates**: Soft updates (τ=0.005) outperformed periodic syncing
- **Network Architecture**: Duelling DQN showed superior performance on complex environments

---

## 🧪 Key Experiments Conducted

### 1. Algorithm Comparison
- **Vanilla DQN vs Duelling DQN**: 15-20% performance improvement with Duelling architecture
- **Target Network Strategies**: Soft updates vs periodic synchronisation analysis
- **Network Depth Impact**: Single vs two hidden layer comparison

### 2. Hyperparameter Sensitivity Analysis
- **Learning Rate Sweep**: Tested rates from 5e-5 to 1e-2
- **Exploration Strategies**: Epsilon start/end/decay parameter optimisation
- **Batch Size Impact**: Memory efficiency vs learning stability trade-offs

### 3. Environment-Specific Adaptations
- **CartPole Optimisation**: Minimal architecture for fast convergence
- **Lunar Lander Scaling**: Larger networks and replay buffers for complex dynamics
- **Cross-Environment Generalisation**: Parameter transfer analysis

---

## 💡 Technical Insights

### Loss Function Implementation
The DQN loss function implements TD-learning with:
```python
# Target value calculation
expected_state_action_values = rewards + next_state_values * gamma * (1 - dones)
# MSE loss between current Q-values and targets
loss = F.mse_loss(state_action_values, expected_state_action_values)
```

### Duelling DQN Architecture
Innovative separation of state value and action advantage:
```python
# Duelling Q-value computation
Q = value + advantage - advantage.mean()
```

### Experience Replay Optimisation
Efficient sampling strategy with configurable buffer sizes:
- CartPole: 10K-1M experiences
- Lunar Lander: 1M experiences for complex dynamics

---

## 🔧 Technologies & Frameworks

- **Deep Learning**: PyTorch for neural network implementation
- **Environment**: OpenAI Gymnasium for reinforcement learning environments
- **Visualisation**: Matplotlib for training progress and performance analysis
- **Configuration**: YAML-based hyperparameter management
- **Hardware**: Multi-platform support (CUDA/MPS/CPU)

---

## 📈 Results Visualisation

Generated comprehensive performance plots including:
- R100 vs Episode Number for all environments
- Learning rate sensitivity analysis
- Architecture comparison charts
- Hyperparameter ablation studies

*All plots and detailed analysis available in the `/figures` directory*

---

## 🎯 Key Learning Outcomes

1. **Algorithm Understanding**: Deep comprehension of Q-learning, TD-learning, and DQN variants
2. **Hyperparameter Optimisation**: Systematic approach to parameter tuning and validation
3. **Performance Analysis**: Statistical evaluation of learning algorithms across environments
4. **Software Engineering**: Clean, modular implementation with configuration management
5. **Research Methodology**: Experimental design and scientific analysis of results

---

*This project demonstrates advanced reinforcement learning implementation skills, systematic experimental methodology, and the ability to optimise complex machine learning systems for practical applications.*Test update
