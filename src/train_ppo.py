import numpy as np
import tensorflow as tf

from environments.rush_hour_env import RushHourEnv
from models.ppo_agent import PPOAgent
from models.rollout_buffer import RolloutBuffer
from environments.ppo_utils import preprocess_obs


def train(env, agent, buf, steps_per_epoch, epochs, batch_size):
    obs, _ = env.reset()
    obs = preprocess_obs(obs)

    for epoch in range(1, epochs+1):
        for t in range(steps_per_epoch):
            # 1) Forward pass: get logits & value
            obs_tensor = tf.expand_dims(obs, 0)
            pi_logits, v = agent.model(obs_tensor, training=False)
            pi_logits = pi_logits[0]       # shape: (act_dim,)
            v = v[0, 0]                     # scalar

            # 2) Sample action from policy
            probs = tf.nn.softmax(pi_logits).numpy()
            act = np.random.choice(env.action_space.n, p=probs)
            logp = np.log(probs[act] + 1e-8)

            # 3) Step environment
            next_obs, rew, done, trunc, _ = env.step(act)
            next_obs = preprocess_obs(next_obs)

            # 4) Store experience
            buf.store(obs, act, logp, v.numpy(), rew)

            obs = next_obs

            # 5) Handle trajectory end
            terminal = done or trunc
            if terminal or t == steps_per_epoch - 1:
                last_val = 0 if done else agent.model(
                    tf.expand_dims(obs, 0)
                )[1][0, 0].numpy()
                buf.finish_path(last_val)
                obs, _ = env.reset()
                obs = preprocess_obs(obs)

        # After collecting steps_per_epoch timesteps, update policy & value
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf = buf.get()

        # PPO: multiple epochs of SGD
        for _ in range(4):
            idxs = np.random.permutation(buf.max_size)
            for start in range(0, buf.max_size, batch_size):
                end = start + batch_size
                batch = idxs[start:end]
                agent.train_step(
                    obs_buf[batch],
                    act_buf[batch],
                    logp_buf[batch],
                    adv_buf[batch],
                    ret_buf[batch]
                )

        print(f"Epoch {epoch}/{epochs} complete")


if __name__ == "__main__":
    # Hyperparameters
    STEPS_PER_EPOCH = 4000
    EPOCHS = 50
    BATCH_SIZE = 64
    OBS_DIM = 36
    PI_LR = 1e-4
    V_LR = 5e-4

    # Init environment, agent, and buffer
    env = RushHourEnv(num_of_vehicle=4, train=True)
    agent = PPOAgent(OBS_DIM, env.action_space.n,
                     pi_lr=PI_LR, v_lr=V_LR)
    buf = RolloutBuffer(size=STEPS_PER_EPOCH, obs_dim=OBS_DIM)

    # Run training
    train(env, agent, buf, STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE)

    # Save weights for later evaluation/visualization
    agent.model.save_weights("ppo_rushhour_weights.h5")
