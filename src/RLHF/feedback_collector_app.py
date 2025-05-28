import streamlit as st
import os
import json
from pathlib import Path
from collections import defaultdict
import random

TRAJ_DIR = Path("database/trajectories_mlp_policy")
OUTPUT_FILE = "feedback_pairwise.json"


def ensure_dir_exists(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_trajectory_pairs(folder):
    folder = ensure_dir_exists(folder)
    try:
        files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
        if not files:
            st.error(
                f"No JSON files found in {folder}. Please add trajectory files first.")
            st.stop()

        by_board = defaultdict(dict)
        for f in files:
            parts = f.split("_")
            if len(parts) >= 3 and parts[0] == "trajectory":
                board_id = parts[1]
                agent_id = parts[2].split(".")[0]
                by_board[board_id][agent_id] = f

        pairs = []
        board_ids = []
        items = list(sorted(by_board.items(), key=lambda x: int(x[0])))

        for board_id, agents in items:
            if "agent1" in agents and "agent2" in agents:
                pairs.append((agents["agent1"], agents["agent2"]))
                board_ids.append(board_id)
        return pairs, board_ids, by_board

    except Exception as e:
        st.error(f"Error loading trajectory files: {str(e)}")
        st.stop()


def load_actions(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        actions = []
        for step in data.get("steps", []):
            action_num = step.get("action")
            vehicle_idx = action_num // 4
            move_idx = action_num % 4
            move_str = ["Up", "Down", "Left", "Right"][move_idx]
            actions.append(f"Vehicle {vehicle_idx}: {move_str}")
        return actions
    except Exception as e:
        st.error(f"Error loading actions from {file_path}: {str(e)}")
        return []


def get_video_bytes(video_path: Path):
    if video_path.exists():
        with open(video_path, "rb") as f:
            return f.read()
    return None


# 🚘 Streamlit UI
st.set_page_config(page_title="Rush Hour Feedback Collector", layout="wide")
st.title("🚗 Rush Hour Trajectory Feedback Collector")

pairs, board_ids, by_board = load_trajectory_pairs(TRAJ_DIR)

if "index" not in st.session_state:
    st.session_state.index = 0
if "results" not in st.session_state:
    st.session_state.results = []

# Display statistics and progress
total_boards = len(by_board)
total_comparisons = len(pairs)
completed_comparisons = len(st.session_state.results)

col_stats1, col_stats2, col_stats3 = st.columns(3)
with col_stats1:
    st.metric("Total Boards", total_boards)
with col_stats2:
    st.metric("Total Comparisons", total_comparisons)
with col_stats3:
    st.metric("Completed Comparisons", completed_comparisons)

# Progress bar
progress = min(1.0, completed_comparisons /
               total_comparisons if total_comparisons > 0 else 0)
st.progress(progress)

if st.session_state.index >= len(pairs):
    st.success("✅ All comparisons completed!")

    # Show summary of completed comparisons
    st.subheader("Summary of Completed Comparisons")
    results_df = defaultdict(lambda: {"agent1_wins": 0, "agent2_wins": 0})
    for result in st.session_state.results:
        board_id = result["trajectory_A"].split("_")[1]
        winner = "agent1_wins" if result["preferred"].endswith(
            "agent1.json") else "agent2_wins"
        results_df[board_id][winner] += 1

    st.write("Results by board:")
    for board_id, stats in sorted(results_df.items()):
        st.write(
            f"Board {board_id}: Agent 1 wins: {stats['agent1_wins']}, Agent 2 wins: {stats['agent2_wins']}")

    if st.button("Download Results"):
        output_path = ensure_dir_exists(Path("database")) / OUTPUT_FILE
        with open(output_path, "w") as f:
            json.dump(st.session_state.results, f, indent=2)
        st.download_button("📥 Download JSON", file_name=OUTPUT_FILE, data=json.dumps(
            st.session_state.results, indent=2))
    st.stop()

# Display current comparison
current_board = board_ids[st.session_state.index]
st.subheader(
    f"🎮 Board {current_board} - Comparison {st.session_state.index + 1} of {len(pairs)}")

fileA, fileB = pairs[st.session_state.index]
pathA = TRAJ_DIR / fileA
pathB = TRAJ_DIR / fileB
videoA = get_video_bytes(pathA.with_name(pathA.stem + "_video.mp4"))
videoB = get_video_bytes(pathB.with_name(pathB.stem + "_video.mp4"))
actionsA = load_actions(pathA)
actionsB = load_actions(pathB)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### Trajectory A\n**File:** {fileA}")
    if videoA:
        st.markdown(
            """
            <style>
            .stVideo {
                width: 100%;
                max-width: 400px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.video(videoA)
    else:
        st.warning("Video not found")
    st.text(f"{len(actionsA)} moves:\n" + "\n".join(actionsA))

with col2:
    st.markdown(f"### Trajectory B\n**File:** {fileB}")
    if videoB:
        st.markdown(
            """
            <style>
            .stVideo {
                width: 100%;
                max-width: 400px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.video(videoB)
    else:
        st.warning("Video not found")
    st.text(f"{len(actionsB)} moves:\n" + "\n".join(actionsB))

option = st.radio("Which trajectory is better?",
                  options=["A", "B"], horizontal=True)

# Navigation controls
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("⬅️ Previous", disabled=st.session_state.index == 0):
        st.session_state.index -= 1
        st.experimental_rerun()

with col_nav2:
    if st.button("Submit Feedback", type="primary"):
        winner = fileA if option == "A" else fileB
        st.session_state.results.append({
            "trajectory_A": fileA,
            "trajectory_B": fileB,
            "preferred": winner
        })
        st.session_state.index += 1
        st.experimental_rerun()

with col_nav3:
    if st.button("Skip ➡️"):
        st.session_state.index += 1
        st.experimental_rerun()

# Show completion status for each board
st.subheader("Board Completion Status")
completion_status = defaultdict(lambda: {"total": 0, "completed": 0})

# Calculate totals
for board_id in board_ids:
    completion_status[board_id]["total"] += 1

# Calculate completed
for result in st.session_state.results:
    board_id = result["trajectory_A"].split("_")[1]
    completion_status[board_id]["completed"] += 1

# Display as a grid
cols = st.columns(5)
for idx, (board_id, status) in enumerate(sorted(completion_status.items())):
    col_idx = idx % 5
    with cols[col_idx]:
        progress = status["completed"] / status["total"]
        st.write(f"Board {board_id}")
        st.progress(progress)
        st.write(f"{status['completed']}/{status['total']}")
