from .python_agent_interface import PythonAgent
from .a2rl.symbol_manager import SymbolManager
from .a2rl.atomic_policy_manager import AtomicPolicyManager
from .a2rl.a2rl_test import A2RLTest
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
        self.conf = conf
        self.behavior:Behavior = Behavior.Test
        # 2个基础
        self.symbol_manager = SymbolManager()
        self.atomic_policy_manager = AtomicPolicyManager(conf)
        # 3个不同处理器
        self.test_handler = A2RLTest(self.conf, self, self.atomic_policy_manager, self.symbol_manager)


    def act(self, obs: np.ndarray[int, np.dtype[np.int_]], pre_reward:int, pre_terminated:bool):
        if self.behavior == Behavior.Test:
            return self.test_handler.act(obs, pre_reward, pre_terminated)
            pass
        elif self.behavior == Behavior.Discover:
            pass
        elif self.behavior == Behavior.Train:
            pass
        else:
            logger.error(f"Invalid a2rl state:{self.state}")


    def changhe_state(self, state: str):
        if state == "test":
            self.behavior = Behavior.Test
        elif state == "discover":
            self.behavior = Behavior.Discover
        elif state == "train":
            self.behavior = Behavior.Train
        else:
            logger.error(f"Invalid a2rl state:{state}")
