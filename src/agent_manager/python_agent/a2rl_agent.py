from .python_agent_interface import PythonAgent
from .a2rl.symbol_manager import SymbolManager
from enum import Enum
import numpy as np
from loguru import logger

class Behavior(Enum):
    Test = 0
    Discover = 1
    Train = 2


class MyAgent(PythonAgent):
    def __init__(self, conf: dict):
        super().__init__(conf)
        self.behavior:Behavior = Behavior.Test
        self.symbol_manager = SymbolManager()

    def act(self, obs: np.ndarray[int, np.dtype[np.int_]]):
        if self.behavior == Behavior.Test:
            pass
        elif self.behavior == Behavior.Discover:
            pass
        elif self.behavior == Behavior.Train:
            pass
        else:
            logger.error(f"Invalid a2rl state:{self.state}")
