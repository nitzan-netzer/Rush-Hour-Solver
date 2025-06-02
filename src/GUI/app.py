import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import time
import numpy as np
from environments.rush_hour_env import RushHourEnv
from PIL import Image, ImageDraw

st.set_page_config(page_title="Rush Hour Solver", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

letter_to_color = {
    "X": (255, 0, 0),       # Red
    "A": (144, 238, 144),   # Light Green
    "B": (255, 165, 0),     # Orange
    "C": (0, 255, 255),     # Cyan
    "D": (255, 182, 193),   # Pink
    "E": (0, 0, 139),       # Dark Blue
    "F": (0, 128, 0),       # Green
    "G": (50, 50, 50),      # Gray
    "H": (245, 245, 220),   # Beige
    "I": (255, 255, 128),   # Light Yellow
    "J": (139, 69, 19),     # Saddle brown
    "K": (0, 255, 0),       # Green
    "O": (255, 255, 0),     # Yellow
    "P": (128, 0, 128),     # Purple
    "Q": (0, 0, 255),       # Blue
    "R": (0, 128, 128),     # Teal
    "": (220, 220, 220),    # Empty space
}

def generate_board_image(board_matrix, cell_size=40):
    rows, cols = board_matrix.shape
    img = Image.new("RGB", (cols * cell_size, rows * cell_size), color="black")
    draw = ImageDraw.Draw(img)

    for i in range(rows):
        for j in range(cols):
            cell = board_matrix[i][j]
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            color = letter_to_color.get(cell, (100, 100, 255))
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="white")
            if cell:
                draw.text((x0 + 12, y0 + 10), cell, fill="white")
    return img.resize((250, 250))

if "started" not in st.session_state:
    st.session_state.started = False
if "env" not in st.session_state:
    st.session_state.env = None
if "initial_img" not in st.session_state:
    st.session_state.initial_img = None

st.title("🚗 Rush Hour Puzzle Solver")
num_vehicles = st.sidebar.slider("Number of vehicles", 3, 6, 6)
max_steps = st.sidebar.slider("Maximum steps", 10, 100, 50)
step_delay = st.sidebar.slider("Animation speed (sec)", 0.05, 1.0, 0.2)

if st.sidebar.button("🚦 Start"):
    st.session_state.env = RushHourEnv(num_of_vehicle=num_vehicles, train=False)
    obs, _ = st.session_state.env.reset()
    board_matrix = st.session_state.env.board.board
    st.session_state.initial_img = generate_board_image(board_matrix)
    st.session_state.started = True

if st.session_state.started and st.session_state.initial_img:
    st.markdown("### 🧩 Your Board")
    st.image(st.session_state.initial_img, caption="Initial Board", use_container_width=False)

if st.session_state.started:
    if st.button("🔍 Solve Now"):
        with st.spinner("Solving..."):
            env = st.session_state.env
            images = []
            rewards = []
            solved = False

            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, done, truncated, _ = env.step(action)

                board_matrix = env.board.board
                img = generate_board_image(board_matrix)
                images.append(img)
                rewards.append(reward)

                if done:
                    solved = True
                    break

            st.markdown(f"## ✅ Solved: {'Yes' if solved else 'No'} in {len(images)} steps")
            st.markdown(f"### 🧮 Total Reward: `{sum(rewards):.2f}`")

            st.markdown("### 🎞️ Animation")
            img_placeholder = st.empty()
            for i, img in enumerate(images):
                img_placeholder.image(img, caption=f"Step {i+1} | Reward: {rewards[i]:.2f}", use_container_width=False)
                time.sleep(step_delay)

            st.markdown("### 🧩 All Steps Grid")
            cols = st.columns(5)
            for i, img in enumerate(images):
                with cols[i % 5]:
                    st.image(img, caption=f"Step {i+1}", use_container_width=False)
