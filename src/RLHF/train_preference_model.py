import os
import json
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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
        trajA = load_state_sequence(pathA)
        trajB = load_state_sequence(pathB)
        label = 1.0 if entry["preferred"] == entry["trajectory_A"] else 0.0
        comparisons.append((trajA, trajB, label))
    return comparisons


def create_model(input_dim):
    """Create a simple feedforward network that outputs a scalar preference score."""
    input_layer = layers.Input(shape=(input_dim + 1,))
    x = layers.Dense(64, activation='relu')(input_layer)
    output = layers.Dense(1)(x)  # Output is a logit (raw score)
    return models.Model(inputs=input_layer, outputs=output)


def train_model(model, comparisons, epochs=100, lr=1e-3):
    """Train the model using a preference-based loss."""
    optimizer = optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        random.shuffle(comparisons)
        total_loss = 0.0

        for trajA, trajB, label in comparisons:
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
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


def main():
    print("🔄 Loading comparisons...")
    comparisons = load_comparisons()

    print("🧠 Creating model...")
    model = create_model(INPUT_DIM)

    print("🏋️ Training preference model...")
    train_model(model, comparisons)

    model.save("preference_model.keras")
    print("✅ Model saved to 'preference_model.keras'")


if __name__ == "__main__":
    main()
