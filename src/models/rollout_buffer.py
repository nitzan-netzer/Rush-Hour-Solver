import numpy as np


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vector x.
    y[t] = x[t] + discount * x[t+1] + discount^2 * x[t+2] + ...
    """
    return np.array([
        sum(discount**i * x[i+j] for i in range(len(x)-j))
        for j in range(len(x))
    ], dtype=np.float32)


class RolloutBuffer:
    def __init__(self, size, obs_dim):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size,       dtype=np.int32)
        self.logp_buf = np.zeros(size,       dtype=np.float32)
        self.val_buf = np.zeros(size,       dtype=np.float32)
        self.ret_buf = np.zeros(size,       dtype=np.float32)
        self.adv_buf = np.zeros(size,       dtype=np.float32)
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, logp, val, rew):
        """
        Save one timestep of data into the buffer.
        """
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.logp_buf[idx] = logp
        self.val_buf[idx] = val
        self.ret_buf[idx] = rew
        self.ptr += 1

    def finish_path(self, last_val, gamma=0.99, lam=0.95):
        """
        Call this at the end of a trajectory (or when epoch ends).
        Computes GAE advantage & discounted returns.
        """
        slice_ = slice(0, self.ptr)
        rews = np.append(self.ret_buf[slice_], last_val)
        vals = np.append(self.val_buf[slice_], last_val)

        # TD residuals
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        # GAE-Lambda advantage
        self.adv_buf[slice_] = discount_cumsum(deltas, gamma * lam)
        # Discounted returns
        self.ret_buf[slice_] = discount_cumsum(rews, gamma)[:-1]

    def get(self):
        """
        Returns all data and resets pointer.
        Normalizes advantages to mean=0, std=1.
        """
        assert self.ptr == self.max_size, "Buffer has to be full before you can get"
        self.ptr = 0
        # Normalize advantages
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return (
            self.obs_buf,
            self.act_buf,
            self.logp_buf,
            self.adv_buf,
            self.ret_buf
        )
