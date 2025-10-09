from ..agent_interface import AgentInterface
import numpy as np


class PythonAgent(AgentInterface):
    def __init__(self, conf: dict):
        super().__init__()

    def act(self, obs: np.ndarray, pre_reward:int, pre_terminated:bool):
        return np.random.randint(0, 11)
