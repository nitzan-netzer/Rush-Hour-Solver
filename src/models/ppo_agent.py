import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class PPOAgent:
    def __init__(self, obs_dim, act_dim,
                 clip_ratio=0.2, pi_lr=1e-4, v_lr=5e-4, ent_coef=0.0):
        # Build actor‑critic network
        self.model = self.build_actor_critic(obs_dim, act_dim)
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef

        # Optimizers with gradient clipping
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr, clipnorm=0.5)
        self.v_optimizer = tf.keras.optimizers.Adam(v_lr,  clipnorm=0.5)

        # Split trainable variables between policy & value heads + shared body
        # model.layers = [Input, Dense_1, Dense_2, PolicyHead, ValueHead]
        shared_layers = self.model.layers[1:3]
        policy_layer = self.model.layers[3]
        value_layer = self.model.layers[4]

        shared_vars = []
        for layer in shared_layers:
            shared_vars += layer.trainable_variables

        self.policy_vars = policy_layer.trainable_variables + shared_vars
        self.value_vars = value_layer.trainable_variables + shared_vars

    def build_actor_critic(self, obs_dim, act_dim):
        x_in = Input(shape=(obs_dim,), dtype='float32')
        x = layers.Dense(64, activation='relu')(x_in)
        x = layers.Dense(64, activation='relu')(x)
        pi_logits = layers.Dense(act_dim, name='pi')(x)
        v = layers.Dense(1,       name='v')(x)
        return Model(x_in, [pi_logits, v])

    @tf.function
    def train_step(self, obs, act, old_logp, adv, ret):
        # Detach targets & advantages from any upstream graph
        old_logp = tf.stop_gradient(old_logp)
        adv = tf.stop_gradient(adv)
        ret = tf.stop_gradient(ret)

        with tf.GradientTape(persistent=True) as tape:
            pi_logits, v = self.model(obs, training=True)
            v = tf.squeeze(v, axis=1)

            # Compute log-probs of taken actions
            logp_all = tf.nn.log_softmax(pi_logits)
            logp = tf.reduce_sum(
                tf.one_hot(act, pi_logits.shape[-1]) * logp_all,
                axis=1
            )

            # PPO clipped objective
            ratio = tf.exp(logp - old_logp)
            min_adv = tf.where(
                adv > 0,
                (1 + self.clip_ratio) * adv,
                (1 - self.clip_ratio) * adv
            )
            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

            # Value (critic) loss
            v_loss = tf.reduce_mean((ret - v)**2)

            # Entropy bonus for exploration
            ent = tf.reduce_mean(
                -tf.reduce_sum(tf.nn.softmax(pi_logits) * logp_all, axis=1)
            )

        # Compute & clip gradients
        pi_grads = tape.gradient(pi_loss, self.policy_vars)
        pi_grads, _ = tf.clip_by_global_norm(pi_grads, 0.5)

        v_grads = tape.gradient(v_loss, self.value_vars)
        v_grads, _ = tf.clip_by_global_norm(v_grads, 0.5)

        # Apply updates
        self.pi_optimizer.apply_gradients(zip(pi_grads, self.policy_vars))
        self.v_optimizer.apply_gradients(zip(v_grads,  self.value_vars))

        return pi_loss, v_loss, ent
