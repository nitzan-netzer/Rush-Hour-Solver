import os
import json
import numpy as np
import random
import tensorflow as tf
import setup_path
from tensorflow.keras import layers, models, optimizers
from utils.config import MODEL_DIR
from datetime import datetime

# === CONFIGURATION ===
TRAJ_DIR = "database/trajectories_mlp_policy"
FEEDBACK_PATH = os.path.join("database", "feedback_pairwise.json")
INPUT_DIM = 36  # Number of cells in a 6x6 Rush Hour board
MAX_LEN = 200  # Max number of steps used to normalize trajectory length


def load_state_sequence(path):
    with open(path) as f:
        data = json.load(f)
    return [step["state"] for step in data["steps"]]


def load_comparisons():
    """Load human preferences and trajectory data into a list of (trajA, trajB, label)."""
    with open(FEEDBACK_PATH) as f:
        feedback = json.load(f)

    comparisons = []
    for entry in feedback:
        pathA = os.path.join(TRAJ_DIR, entry["trajectory_A"])
        pathB = os.path.join(TRAJ_DIR, entry["trajectory_B"])
        try:
            trajA = load_state_sequence(pathA)
            trajB = load_state_sequence(pathB)

            # Skip if either trajectory is empty
            if not trajA or not trajB:
                print(
                    f"⚠️ Skipping empty trajectory: {pathA if not trajA else pathB}")
                continue

            label = 1.0 if entry["preferred"] == entry["trajectory_A"] else 0.0
            comparisons.append((trajA, trajB, label))
        except Exception as e:
            print(
                f"⚠️ Error loading trajectories {pathA} or {pathB}: {str(e)}")
            continue

    if not comparisons:
        raise ValueError("No valid comparisons found in the dataset!")

    print(f"✅ Loaded {len(comparisons)} valid comparisons")
    return comparisons


def create_model(input_dim):
    input_layer = layers.Input(shape=(input_dim + 1,))
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=input_layer, outputs=output)


def train_model(model, comparisons, epochs=100, lr=1e-3):
    """Train the model using a preference-based loss."""
    if not comparisons:
        raise ValueError("No comparisons provided for training!")

    optimizer = optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        random.shuffle(comparisons)
        total_loss = 0.0
        valid_comparisons = 0

        for trajA, trajB, label in comparisons:
            try:
                # Validate trajectory data
                if not trajA or not trajB:
                    continue

                finalA = np.array(trajA[-1], dtype=np.float32).reshape(1, -1)
                finalB = np.array(trajB[-1], dtype=np.float32).reshape(1, -1)
                stepsA = np.array([[len(trajA) / MAX_LEN]], dtype=np.float32)
                stepsB = np.array([[len(trajB) / MAX_LEN]], dtype=np.float32)
                label_tensor = np.array([[label]], dtype=np.float32)

                with tf.GradientTape() as tape:
                    inputA = tf.concat([finalA, stepsA], axis=1)
                    inputB = tf.concat([finalB, stepsB], axis=1)

                    scoreA = model(inputA, training=True)
                    scoreB = model(inputB, training=True)
                    score_diff = scoreA - scoreB

                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=score_diff,
                        labels=label_tensor
                    ))

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                total_loss += loss.numpy()
                valid_comparisons += 1

            except Exception as e:
                print(f"⚠️ Error processing comparison: {str(e)}")
                continue

        if valid_comparisons > 0:
            avg_loss = total_loss / valid_comparisons
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} ({valid_comparisons} valid comparisons)")
        else:
            print(
                f"⚠️ Epoch {epoch+1}/{epochs} - No valid comparisons processed")


def main():
    print("🔄 Loading comparisons...")
    comparisons = load_comparisons()

    print("🧠 Creating model...")
    model = create_model(INPUT_DIM)

    print("🏋️ Training preference model...")
    train_model(model, comparisons)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(MODEL_DIR / f"preference_model_{timestamp}.keras")
    print(f"✅ Model saved to 'preference_model_{timestamp}.keras'")


if __name__ == "__main__":
    main()
