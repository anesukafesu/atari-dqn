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


def play():
    # Start virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    env = gym.make("ALE/Pacman-v5", render_mode="rgb_array")

    # Setup the wrapper to record the video
    video_callable=lambda episode_id: True
    env = RecordVideo(env, video_folder='./videos', episode_trigger=video_callable)
    
    # Load trained model
    model = DQN.load("/kaggle/working/dqn_model2.zip")

    obs, _ = env.reset()
    done = False
    truncated = False

    while not(done or truncated):
        obs = np.array(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
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
