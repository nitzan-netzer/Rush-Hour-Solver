import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_logs(log_file_path):
    """Analyze the logs and generate plots based on the given log file."""
    log_file = Path(log_file_path)

    # === Load the CSV ===
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return

    df = pd.read_csv(log_file)

    # === Basic stats ===
    print("\n📊 Training Summary:")
    print(f"Total episodes: {len(df)}")
    print(f"Average reward: {df['reward'].mean():.2f}")
    print(f"Max reward: {df['reward'].max():.2f}")
    print(f"Red car escape rate: {df['red_car_escaped'].mean() * 100:.1f}%")

    # === Plot: Reward per Episode ===
    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["reward"], label="Reward", alpha=0.7)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot: Red Car Escaped (Binary) ===
    plt.figure(figsize=(10, 3))
    plt.plot(
        df["episode"],
        df["red_car_escaped"],
        label="Red Car Escaped (1=True)",
        color="green",
        alpha=0.6,
    )
    plt.title("Red Car Escape Events")
    plt.xlabel("Episode")
    plt.ylabel("Escaped")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Rolling Average Reward (optional) ===
    if "reward" in df.columns:
        df["rolling_reward"] = df["reward"].rolling(window=20).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(
            df["episode"],
            df["rolling_reward"],
            label="Rolling Avg (20)",
            color="orange",
        )
        plt.title("Smoothed Reward (Rolling Average)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("✅ Log analysis completed. Plots displayed.")
