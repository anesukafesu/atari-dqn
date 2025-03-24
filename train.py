#!/usr/bin/env python3
""" Trains an DQN Agent with CNN Policy
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Get latest episode rewards
        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            ep_length = self.locals["infos"][0]["episode"]["l"]

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)

            # Log to TensorBoard
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-10:]))  # Moving avg
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-10:]))

        return True


def train():
  # Define the environment
  gym.register_envs(ale_py)
  environment = gym.make('ALE/Pacman-v5', render_mode='rgb_array')
  environment = AtariWrapper(environment) # Implements preprocessing steps such as grayscale and frame skipping
  environment = DummyVecEnv([lambda: environment]) # Vectorisation enables training multiple agents at once?
  environment = VecFrameStack(environment, n_stack=4) # Stacks 4 frames for temporal awareness

  # Define the agent
  model = DQN(
    'CnnPolicy',
    environment,
    verbose=1,
    tensorboard_log='./dqn_pacman_log/',
    learning_rate=0.00001,
    buffer_size=100000,
    batch_size=32,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    train_freq=4,
    target_update_interval=1000,
    gamma=0.85,
    device='cuda'
  )

  # Train the model
  model.learn(total_timesteps=1000000, callback=RewardLoggingCallback())

  # Save the model
  model.save(f'dqn_model.zip')


if __name__ == "__main__":
    train()