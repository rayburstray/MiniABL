import numpy as np
from abc import ABC, abstractmethod

class AgentInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """
        :param obs: observation(此obs为符号化的而非原始像素，当然若rec为none的话，obs为原始像素)
        :return: action
        """
        pass
