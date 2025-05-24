import streamlit as st
import os
import json
from pathlib import Path

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
        pairs = []
        for i in range(0, len(files) - 1, 2):
            pairs.append((files[i], files[i + 1]))
        return pairs
    except Exception as e:
        st.error(f"Error loading trajectory files: {str(e)}")
        st.stop()


def load_actions(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract actions from steps
        actions = []
        for step in data.get("steps", []):
            action_num = step.get("action")
            # Convert action number to move description
            vehicle_idx = action_num // 4
            move_idx = action_num % 4
            move_str = ["Up", "Down", "Left", "Right"][move_idx]
            actions.append(f"Vehicle {vehicle_idx}: {move_str}")

        return actions
    except Exception as e:
        st.error(f"Error loading actions from {file_path}: {str(e)}")
        return []


def get_video_path(json_path):
    # Convert trajectory_X_agentY.json to trajectory_X_agentY_video.mp4
    return json_path.parent / (json_path.stem + "_video.mp4")


st.title("🚗 Rush Hour Trajectory Feedback Collector")

# Load trajectory pairs
pairs = load_trajectory_pairs(TRAJ_DIR)
if "index" not in st.session_state:
    st.session_state.index = 0
if "results" not in st.session_state:
    st.session_state.results = []

# Check for bounds
if st.session_state.index >= len(pairs):
    st.success("✅ All comparisons completed!")
    if st.button("Download Results"):
        output_path = ensure_dir_exists(Path("database")) / OUTPUT_FILE
        with open(output_path, "w") as f:
            json.dump(st.session_state.results, f, indent=2)
        st.download_button("📥 Download JSON", file_name=OUTPUT_FILE, data=json.dumps(
            st.session_state.results, indent=2))
    st.stop()

fileA, fileB = pairs[st.session_state.index]
actionsA = load_actions(TRAJ_DIR / fileA)
actionsB = load_actions(TRAJ_DIR / fileB)

st.subheader(f"🆚 Comparison {st.session_state.index + 1} of {len(pairs)}")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### Trajectory A\n**File:** {fileA}")
    video_path_a = get_video_path(TRAJ_DIR / fileA)
    if video_path_a.exists():
        st.video(str(video_path_a))
    else:
        st.warning("Video not found")
    st.text(f"{len(actionsA)} moves:\n" + "\n".join(actionsA))

with col2:
    st.markdown(f"### Trajectory B\n**File:** {fileB}")
    video_path_b = get_video_path(TRAJ_DIR / fileB)
    if video_path_b.exists():
        st.video(str(video_path_b))
    else:
        st.warning("Video not found")
    st.text(f"{len(actionsB)} moves:\n" + "\n".join(actionsB))

option = st.radio("Which trajectory is better?",
                  options=["A", "B"], horizontal=True)

if st.button("Submit Feedback"):
    winner = fileA if option == "A" else fileB
    st.session_state.results.append({
        "trajectory_A": fileA,
        "trajectory_B": fileB,
        "preferred": winner
    })
    st.session_state.index += 1
    st.experimental_rerun()
