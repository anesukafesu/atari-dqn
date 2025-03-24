# Deep Q-Learning Pacman Agent

## Project Overview

This project implements a Deep Q-Network (DQN) agent to play the Atari Pacman game using Stable Baselines3 and Gymnasium. The goal was to train an intelligent agent that can navigate and succeed in the Pacman environment through reinforcement learning.

## Environment

- **Game**: Atari Pacman (ALE/Pacman-v5)
- **Learning Algorithm**: Deep Q-Network (DQN)
- **Policy Network**: Convolutional Neural Network (CNN Policy)

## Hyperparameter Exploration

### Experimental Configurations and Performance Analysis

| Experiment                 | Gamma (γ) | Learning Rate (α) | Epsilon Parameters                      | Noted Observations                                                                                                                                                                                                                                                                                                                |
| -------------------------- | --------- | ----------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1: Long-Term Focus         | 0.95      | 0.00005           | Start: 1.0<br>End: 0.01<br>Decay: 0.05  | <br>- High gamma encouraged long-term thinking<br>- Faster learning rate caused instability<br>- Exploration was too structured, limiting random discoveries<br>- Agent showed promise but lacked adaptability                                                                                                                    |
| 2: Faster Learning         | 0.80      | 0.0001            | Start: 1.0<br>End: 0.05<br>Decay: 0.02  | <br>- Low gamma (0.80) made agent too short-sighted<br>- Very high learning rate caused erratic weight updates<br>- Rapid exploration decay limited learning opportunities<br>- Agent struggled to develop a coherent strategy                                                                                                    |
| 3: Extended Exploration    | 0.90      | 0.00001           | Start: 1.0<br>End: 0.001<br>Decay: 0.1  | <br>- Balanced gamma (0.90) enables strategic thinking<br>- Moderate learning rate ensures stable updates<br>- Extremely slow exploration decay allows thorough environment understanding<br>- Agent can explore extensively and refine strategy gradually<br>- Lowest epsilon end (0.001) maintains exploratory behavior longest |
| 4: Aggressive Exploitation | 0.85      | 0.00001           | Start: 0.5<br>End: 0.01<br>Decay: 0.005 | <br>- Started with reduced exploration (0.5)<br>- Very fast decay limited learning opportunities<br>- Agent quickly locked into suboptimal strategy<br>- Reduced initial randomness prevented discovering better approaches                                                                                                       |

### Why Experiment 3 Succeeded

#### Key Success Factors

1. **Balanced Long-Term Thinking**

   - Gamma of 0.90 strikes an optimal balance between immediate and future rewards
   - Allows agent to consider consequences beyond immediate actions
   - Encourages more strategic decision-making in Pacman environment

2. **Extensive Exploration**

   - Extremely slow exploration decay (0.1)
   - Epsilon end of 0.001 ensures continued exploration
   - Enables agent to continuously discover and refine strategies
   - Prevents premature convergence to a suboptimal policy

3. **Stable Learning**
   - Conservative learning rate (0.00001)
   - Prevents sudden, disruptive weight updates
   - Allows gradual, stable improvement of policy
   - Maintains consistency in learning process

#### Pacman-Specific Advantages

- Allows agent to learn complex maze navigation
- Discovers nuanced ghost-avoiding strategies
- Develops more adaptive approach to collecting points
- Maintains curiosity about environment throughout training

## Key Components

### train.py

- Implements DQN training using Stable Baselines3
- Utilizes AtariWrapper for preprocessing
- Applies frame stacking for temporal awareness
- Logs reward and episode length metrics

### play.py

- Loads trained model
- Replays agent's performance
- Records gameplay video
- Demonstrates learned policy

## Prerequisites

- Python 3.8+
- Libraries:
  - stable-baselines3
  - gymnasium
  - ale-py
  - opencv-python
  - numpy

## Training Process

1. Environment preprocessing
2. DQN agent initialization
3. Training for 1,000,000 timesteps
4. Model saving and evaluation

## Results Visualization

Gameplay video demonstrates the agent's improved performance using the optimized hyperparameters, showcasing:

- More strategic movement
- Better understanding of game mechanics
- Increased reward collection

Additional logs on reward and loss are stored in the `tensorboard_logs` folder.

## Gameplay Video

[Watch the Pacman AI in Action](https://github.com/anesukafesu/atari-dqn/raw/main/formative.mp4)

## Lessons Learned

- Critical role of hyperparameter tuning
- Balancing exploration and long-term reward consideration
- Importance of gradual exploration decay

## Contributions

- Emmanuel Yakubu (Training and Documentation)
- Joak (Training)
- Anesu (Training and Playing)
