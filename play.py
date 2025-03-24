#!/usr/bin/env python3
""" Plays an Atari Game using the presaved model.
    This script should be executed in a Jupyter Notebook Environment.
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo, TransformObservation, FrameStack
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import cv2
import numpy as np


def preprocess_observation(obs):
    obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
    obs_resized = obs_resized / 255.0
    return np.array(obs_resized, dtype=np.float32)


def play():
    # Start virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    env = gym.make("ALE/Pacman-v5", render_mode="rgb_array")
    env = TransformObservation(env, preprocess_observation)
    env = FrameStack(env, num_stack=4)

    # Setup the wrapper to record the video
    video_callable=lambda _: True
    env = RecordVideo(env, video_folder='./videos', episode_trigger=video_callable)
    
    # Load trained model
    model = DQN.load("dqn_model.zip")

    obs, _ = env.reset()
    done = False
    truncated = False

    while not(done or truncated):
        obs = np.array(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        env.render()

        if done or truncated:
            obs, _ = env.reset()

    env.close()

    # Display the video
    video = io.open(glob.glob('videos/*.mp4')[-1], 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>
    '''.format(encoded.decode('ascii'))))

if __name__ == "__main__":
    play()