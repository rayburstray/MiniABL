import numpy as np


class AgentInterface:
    def __init__(self):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        pass
