# DQN variants in PyTorch

Deep Q-Network agents for CartPole and Lunar Lander (Gymnasium), with a
hyperparameter study of each. Built for UQ COMP3702 (Artificial Intelligence).

It compares three network architectures: a single hidden layer, two hidden
layers, and a duelling network that estimates state value and action advantage
separately. All three use experience replay and a target network, synced
either periodically or by soft updates.

## Results

R100 is the mean reward over the last 100 episodes.

| Environment | Target R100 | Achieved R100 | Episodes (approx.) | Best architecture |
|---|---|---|---|---|
| CartPole-v0 | 195.0 | 196.32 | 500 | Single hidden layer |
| CartPole-v1 | 475.0 | 475.83 | 1,200 | Duelling |
| LunarLander-v3 | 200.0 | 202.98 | 2,500 | Duelling |

The study swept learning rates from 5e-5 to 1e-2 and compared periodic target
syncing with soft updates (τ = 0.005), along with epsilon schedules and network
depth. Training curves for each run are in [`figures/`](figures).

## Layout

```
dqn_gym.py                 Training loop and environment interface
dqn_common.py              Network architectures, epsilon schedule, soft update
lib/experience_buffer.py   Experience replay buffer
config/dqn.yaml            Hyperparameters per environment
figures/                   R100 training curves
```

## Running

```bash
conda env create -f environment.yml    # or environment_arm64.yml on Apple silicon
conda activate DQN2                    # DQN2_arm64 for the arm64 environment

python dqn_gym.py -e CartPole-v1 -n single-hidden
python dqn_gym.py -e LunarLander-v3 -n duelling-dqn
python dqn_gym.py -e CartPole-v1 -n duelling-dqn -v    # live R100 plot
```

| Flag | Meaning |
|---|---|
| `-e`, `--env` | `CartPole-v0`, `CartPole-v1` or `LunarLander-v3` |
| `-n`, `--network` | `single-hidden`, `two-hidden` or `duelling-dqn` |
| `-v`, `--live_plot` | Plot R100 live during training |
| `-s`, `--seed` | Seed for reproducibility |
| `-c`, `--config_file` | Hyperparameter file (default `config/dqn.yaml`) |

Training runs on CUDA, Apple MPS or CPU, whichever is available.
