import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_logs(log_files_list):
    """Analyze the logs and generate plots based on the given log file."""
    
    # Dictionary to store DataFrames
    dfs = {}
    
    # Load all log files
    for log_file_path in log_files_list:
        log_file = Path(log_file_path)
        if not log_file.exists():
            print(f"❌ Log file not found: {log_file}")
            continue
            
        df = pd.read_csv(log_file)
        name = log_file.stem  # Get filename without extension
        dfs[name] = df
        
        # Print basic stats for each file
        print(f"\n📊 Training Summary for {name}:")
        print(f"Total episodes: {len(df)}")
        print(f"Average reward: {df['reward'].mean():.2f}")
        print(f"Max reward: {df['reward'].max():.2f}")
        print(f"Red car escape rate: {df['red_car_escaped'].mean() * 100:.1f}%")
    
    # === Plot: Compare Rewards ===
    plt.figure(figsize=(10, 5))
    for name, df in dfs.items():
        plt.plot(df["episode"], df["reward"], label=name, alpha=0.7)
    plt.title("Reward Comparison Across Logs")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # === Plot: Compare Rolling Averages ===
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
    
    # === Plot: Compare Escape Rates ===
    plt.figure(figsize=(10, 5))
    for name, df in dfs.items():
        plt.plot(
            df["episode"],
            df["red_car_escaped"],
            label=name,
            alpha=0.6,
        )
    plt.title("Red Car Escape Events Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Escaped")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("✅ Log analysis completed. Plots displayed.")

# Example usage:
if __name__ == "__main__":
    logs = [
        "logs/rush_hour/example.csv",
    ]
    analyze_logs(logs)
