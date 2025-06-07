from pathlib import Path
MODEL_DIR = Path("models_zip/")  # Directory to store models
LOG_DIR = Path("logs/csv/")  # Directory to store logs
VIDEO_DIR = Path("logs/videos/")  # Directory to store videos

MODEL_PATH = MODEL_DIR / "rush_hour"
LOG_FILE_PATH = LOG_DIR / "run_latest.csv"
VIDEO_PATH = VIDEO_DIR / "rush_hour_demo.mp4"
NUM_VEHICLES = 16

CNN_MODEL_PATH = MODEL_DIR / "rush_hour_cnn"
CNN_LOG_FILE_PATH = LOG_DIR / "run_cnn_latest.csv"
