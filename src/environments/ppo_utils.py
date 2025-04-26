import numpy as np


def preprocess_obs(obs):
    """
    Scale uint8 observations [0,255] → float32 [0,1].
    """
    return obs.astype(np.float32) / 255.0
