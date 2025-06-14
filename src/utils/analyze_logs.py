import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_logs(log_files_list):
    dfs = {}

    for log_file_path in log_files_list:
        log_file = Path(log_file_path)
        if not log_file.exists():
            print(f"❌ Log file not found: {log_file}")
            continue

        df = pd.read_csv(log_file)
        name = log_file.stem
        dfs[name] = df

        print(f"\n📊 Training Summary for {name}:")
        print(f"Total episodes: {len(df)}")
        print(f"Average reward: {df['reward'].mean():.2f}")
        print(f"Max reward: {df['reward'].max():.2f}")
        print(f"Red car escape rate: {df['red_car_escaped'].mean() * 100:.1f}%")

        if "steps_to_solve" in df.columns:
            steps_series = pd.to_numeric(df["steps_to_solve"], errors='coerce').dropna()
            if not steps_series.empty:
                print(f"Average steps to solve: {steps_series.mean():.2f}")
                print(f"Min steps to solve: {steps_series.min():.0f}")
                print(f"Max steps to solve: {steps_series.max():.0f}")

    # === Plot: Steps to Solve Per Episode ===
    plt.figure(figsize=(10, 5))
    for name, df in dfs.items():
        steps = pd.to_numeric(df["steps_to_solve"], errors='coerce')
        plt.plot(df["episode"], steps, label=name, alpha=0.7)
    plt.title("Steps to Solve (Only Escaped Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot: Rolling Avg Reward ===
    plt.figure(figsize=(10, 5))
    for name, df in dfs.items():
        rolling_reward = df["reward"].rolling(window=20).mean()
        plt.plot(df["episode"], rolling_reward, label=f"{name} (Rolling Avg)", alpha=0.7)
    plt.title("Smoothed Reward Comparison (Rolling Average)")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot: Red Car Escape Events ===
    plt.figure(figsize=(10, 5))
    for name, df in dfs.items():
        plt.plot(df["episode"], df["red_car_escaped"], label=name, alpha=0.6)
    plt.title("Red Car Escape Events")
    plt.xlabel("Episode")
    plt.ylabel("Escaped")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("✅ Log analysis completed. Plots displayed.")
