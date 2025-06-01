import setup_path  # Ensure src is in path

from pathlib import Path
from stable_baselines3 import PPO
from environments.rush_hour_image_env import RushHourImageEnv
from environments.rush_hour_env import RushHourEnv
from environments.rewards import basic_reward
from models.RL_model import RLModel
from utils.config import MODEL_DIR, LOG_DIR, NUM_VEHICLES


def train_single_cnn_model(
    model_class=PPO,
    use_cnn=True,
    early_stopping=True,
    run_name="cnn_standalone"
):
    model_type = "CNN" if use_cnn else "MLP"
    stop_type = "early" if early_stopping else "full"
    model_filename = f"{model_class.__name__}_{model_type}_{stop_type}_{run_name}.zip"
    model_path = MODEL_DIR / model_filename
    log_file = LOG_DIR / f"{model_filename}.csv"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if use_cnn:
        env = RushHourImageEnv(
            num_of_vehicle=NUM_VEHICLES,
            train=True,
            rewards=basic_reward,
            image_size=(64, 64)  # Lower resolution saves memory
        )
    else:
        env = RushHourEnv(
            num_of_vehicle=NUM_VEHICLES,
            train=True,
            rewards=basic_reward
        )

    model = RLModel(
        model_class=model_class,
        env=env,
        model_path=model_path,
        log_file=log_file,
        early_stopping=early_stopping,
        cnn=use_cnn
    )

    model.train()
    model.save()

    # Evaluation
    test_env = RushHourImageEnv(NUM_VEHICLES, train=False, rewards=basic_reward, image_size=(64, 64)) \
        if use_cnn else RushHourEnv(NUM_VEHICLES, train=False, rewards=basic_reward)

    model.evaluate(test_env)
    return model_path, log_file


if __name__ == "__main__":
    train_single_cnn_model(
        model_class=PPO,
        use_cnn=True,
        early_stopping=True,
        run_name="cnn_standalone"
    )
