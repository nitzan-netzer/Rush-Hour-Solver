from pathlib import Path

MODEL_PATH = "models/ppo_rush_hour_model"
LOG_FILE = "logs/rush_hour/run_latest.csv"
MODEL_DIR = Path("models/")  # Directory to store models
LOG_DIR = Path("logs/rush_hour/")  # Directory to store logs
VIDEO_DIR = Path("videos/")  # Directory to store videos
VIDEO_PATH = VIDEO_DIR / "rush_hour_demo.mp4"  # Path to save the video
NUM_VEHICLES = 6