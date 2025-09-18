from ..agent_interface import AgentInterface
import numpy as np


class PythonAgent(AgentInterface):
    def __init__(self, conf: dict):
        super().__init__()

    def act(self, obs: np.ndarray):
        return np.random.randint(0, 11)
