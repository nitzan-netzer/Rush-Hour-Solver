import os
import json
import matplotlib.pyplot as plt
import imageio
from collections import defaultdict
from pathlib import Path

TRAJ_DIR = Path("database/trajectories_mlp_policy")


def get_episodes(traj_dir):
    ids = set()
    files = os.listdir(traj_dir)
    for f in files:
        if f.endswith("_agent1.json"):
            ep = f.split("_")[1]
            if (f"trajectory_{ep}_agent2.json" in files and
                f"trajectory_{ep}_agent1_video.mp4" in files and
                    f"trajectory_{ep}_agent2_video.mp4" in files):
                ids.add(ep)
    return sorted(ids, key=int)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_video_length(path):
    try:
        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        return meta.get('duration', reader.count_frames() / meta['fps'])
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


def analyze_trajectories(traj_dir=TRAJ_DIR):
    episodes = get_episodes(traj_dir)
    print(f"Analyzing {len(episodes)} episodes")
    red_car_escapes = defaultdict(int)
    video_lengths = []
    agent_steps = defaultdict(list)
    board_hashes = set()

    for ep in episodes:
        for agent in [1, 2]:
            jpath = traj_dir / f"trajectory_{ep}_agent{agent}.json"
            vpath = traj_dir / f"trajectory_{ep}_agent{agent}_video.mp4"

            try:
                jdata = load_json(jpath)
                print(f"Loaded {jpath.name}")

                board_hashes.add(
                    (jdata.get("board_index"), jdata.get("run_index")))

                if jdata.get("steps") and isinstance(jdata["steps"], list):
                    last_step = jdata["steps"][-1]
                    escaped = jdata.get("red_car_escaped", False)
                    steps = last_step.get("step_num", len(jdata["steps"]))
                else:
                    escaped = any(step.get("done", False)
                                  for step in jdata.get("steps", []))
                    steps = len(jdata.get("steps", []))
            except Exception as e:
                print(f"Failed to parse {jpath}: {e}")
                escaped = False
                steps = 0

            if escaped:
                red_car_escapes[f"agent{agent}"] += 1
            agent_steps[f"agent{agent}"].append(steps)

            dur = get_video_length(vpath)
            if dur is not None:
                video_lengths.append(dur)

    num_eps = len(episodes)
    print(f"\n--- Red Car Escapes ---")
    for agent in [1, 2]:
        print(f"Agent {agent}: {red_car_escapes[f'agent{agent}']} / {
              num_eps} ({red_car_escapes[f'agent{agent}']/num_eps:.1%})")

    # Plot 1: Escapes per agent
    plt.bar(["Agent 1", "Agent 2"], [red_car_escapes["agent1"],
            red_car_escapes["agent2"]], color=["blue", "green"])
    plt.title("Boards Escaped by Red Car (per agent)")
    plt.ylabel("Boards Escaped")
    plt.xlabel("Agent")
    plt.show()

    # Plot 2: Video duration histogram
    plt.hist(video_lengths, bins=20, color='purple', alpha=0.7)
    plt.title("Distribution of Solution Video Lengths")
    plt.xlabel("Video Duration (seconds)")
    plt.ylabel("Number of Videos")
    plt.show()

    # Plot 3: Steps histogram
    plt.hist([agent_steps["agent1"], agent_steps["agent2"]], bins=20,
             stacked=True, label=["Agent 1", "Agent 2"], alpha=0.7)
    plt.title("Distribution of Solution Lengths (Steps)")
    plt.xlabel("Steps in Solution")
    plt.ylabel("Number of Solutions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    analyze_trajectories()
