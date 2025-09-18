import numpy as np
from abc import ABC, abstractmethod


# =============== #
# 视觉模式识别模块 #
# =============== #
class RecInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def recognize(self, obs: np.ndarray) -> np.ndarray:
        pass
