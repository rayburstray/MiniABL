from .atomic_policy_manager import AtomicPolicyManager
from .symbol_manager import SymbolManager
from ..a2rl_agent import MyAgent, Behavior
import numpy as np
import random



class A2RLTrain:
    def __init__(self, conf:dict, a2rl_agent:MyAgent, atomic_policy_manager:AtomicPolicyManager, symbol_manager:SymbolManager):
        self.conf = conf
        self.a2rl_agent = a2rl_agent
        self.atomic_policy_manager = atomic_policy_manager
        self.symbol_manager = symbol_manager


    def act(self, obs: np.ndarray[int, np.dtype[np.int_]], pre_reward:int, pre_terminated:bool) -> int:
        pass