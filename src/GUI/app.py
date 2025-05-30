# src/GUI/app.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import random
from PIL import Image
from stable_baselines3 import PPO

from environments.rush_hour_env import RushHourEnv
from GUI.board_to_image import generate_board_image
from utils.config import MODEL_PATH

# Load model and boards
model = PPO.load(MODEL_PATH)
sample_boards = random.sample(RushHourEnv.test_boards, 10)

st.title("🚗 Rush Hour Solver")

board_idx = st.selectbox("Choose a board:", list(range(len(sample_boards))))
board = sample_boards[board_idx]

st.image(generate_board_image(board, scale=30, draw_letters=True), caption=f"Board {board_idx + 1}")

if st.button("Now solve"):
    env = RushHourEnv(num_of_vehicle=4)
    obs, _ = env.reset(board)

    images = []
    for _ in range(50):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        img = generate_board_image(env.board, scale=30, draw_letters=True)
        images.append(img)
        if done:
            break

    st.success("The model solved the board")
    for img in images:
        st.image(img)
