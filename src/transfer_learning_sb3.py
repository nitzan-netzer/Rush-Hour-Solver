#!/usr/bin/env python3
"""
transfer_learning_sb3.py

Fine‑tune a pretrained 4‑vehicle PPO model on a normalized 5‑vehicle Rush Hour environment,
logging metrics to CSV, plotting them, and evaluating on both train & test boards.
"""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environments.rush_hour_env import RushHourEnv
from environments.evaluate import evaluate_model
from logs_utils.custom_logger import RushHourCSVLogger
from logs_utils.analyze_logs import analyze_logs
from environments.rewards import scaled_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer‑learn PPO from 4 to 5 vehicles in Rush Hour (CSV logging)."
    )
    parser.add_argument(
        "--old_model",
        type=str,
        default="models_zip/ppo_rush_hour_run_1745145973.zip",  # ← fixed slash
        help="Path to the pretrained 4‑vehicle PPO model"
    )
    parser.add_argument(
        "--new_model",
        type=str,
        default="models_zip/ppo_rush_hour_5vehicles_transfer.zip",
        help="Where to save the fine‑tuned 5‑vehicle model"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Number of timesteps for transfer‑learning"
    )
    parser.add_argument(
        "--freeze_extractor",
        action="store_true",
        help="Freeze feature extractor layers during fine‑tuning"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/rush_hour_transfer.csv",
        help="CSV file path for training logs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load pretrained 4‑vehicle model
    print("🔄 Loading pretrained 4‑vehicle model...")
    old_model = PPO.load(args.old_model)

    # 2. Create normalized, vectorized 5‑vehicle environments with scaled rewards
    print("📦 Creating 5‑vehicle environments with VecNormalize & scaled rewards...")
    raw_train = DummyVecEnv([lambda: RushHourEnv(
        num_of_vehicle=5,
        train=True,
        rewards=scaled_reward
    )])
    train_env = VecNormalize(
        raw_train,
        norm_obs=True,
        norm_reward=False,   # do NOT normalize rewards
        clip_obs=10.0,
        gamma=0.99
    )

    raw_test = DummyVecEnv([lambda: RushHourEnv(
        num_of_vehicle=5,
        train=False,
        rewards=scaled_reward
    )])
    test_env = VecNormalize(
        raw_test,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    test_env.training = False  # freeze running stats on test

    # 3. Instantiate fresh PPO for 5 vehicles with tuned hyperparameters
    print("🚀 Initializing new PPO for 5 vehicles...")
    new_model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,    # increased to match smaller reward scale
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.1,       # tighter clipping
        clip_range_vf=0.1,
        ent_coef=0.01,        # small entropy bonus to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,

        gae_lambda=0.95,      # smoother advantage estimates
        target_kl=0.02,       # early stop if KL explodes
        verbose=1,
    )

    # 4. Transfer only matching weights from old extractor
    print("📥 Transferring feature‑extractor weights...")
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    matched = {
        k: v for k, v in old_sd.items()
        if k in new_sd and v.shape == new_sd[k].shape
    }
    new_sd.update(matched)
    new_model.policy.load_state_dict(new_sd)

    # 5. Optionally freeze extractor layers
    if args.freeze_extractor:
        print("⛔ Freezing MLP extractor layers...")
        for name, param in new_model.policy.named_parameters():
            if "mlp_extractor" in name:
                param.requires_grad = False

    # 6. Fine‑tune with CSV logging callback
    log_file = Path(args.log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"🖊️ Logging training to {log_file}")
    csv_logger = RushHourCSVLogger(log_path=str(log_file))

    print(f"🏋️ Fine‑tuning for {args.timesteps:,} timesteps...")
    new_model.learn(
        total_timesteps=args.timesteps,
        callback=csv_logger
    )

    # 7. Save the fine‑tuned model
    print(f"💾 Saving transferred model to {args.new_model}...")
    new_model.save(args.new_model)

    # 8. Evaluate on both splits using raw envs
    print("\n📊 Evaluating on training boards:")
    raw_eval_train = RushHourEnv(
        num_of_vehicle=5, train=True, rewards=scaled_reward)
    evaluate_model(new_model, raw_eval_train)

    print("\n📊 Evaluating on test boards:")
    raw_eval_test = RushHourEnv(
        num_of_vehicle=5, train=False, rewards=scaled_reward)
    evaluate_model(new_model, raw_eval_test)

    # 9. Analyze and plot CSV logs
    print(f"\n📈 Analyzing logs and showing plots from {log_file}...")
    analyze_logs(str(log_file))

    print("\n✅ Transfer‑learning with scaled rewards, VecNormalize, and CSV logging complete!")


if __name__ == "__main__":
    main()
