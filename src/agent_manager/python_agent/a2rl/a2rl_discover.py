from .atomic_policy_manager import AtomicPolicyManager
from .symbol_manager import SymbolManager
from ..a2rl_agent import MyAgent, Behavior
import numpy as np

class A2RLDiscover:
    def __init__(self, conf:dict, a2rl_agent:MyAgent, atomic_policy_manager:AtomicPolicyManager, symbol_manager:SymbolManager):
        self.conf = conf
        self.a2rl_agent = a2rl_agent
        self.atomic_policy_manager = atomic_policy_manager
        self.symbol_manager = symbol_manager
        self.create_flag = False # 是否已经创建过了新的policy
        pass

    # def create_policy(self):
    #     self.atomic_policy_manager.create_policy()
    #     self.new_policy_index = self.atomic_policy_manager.policy_nums - 1
    #     self.new_policy = self.atomic_policy_manager.polocies[self.new_policy_index]
    #     self.create_flag = True
    #     self.symbol_manager.reset()  # 重置符号管理器
    def act(self, obs: np.ndarray[int, np.dtype[np.int_]], pre_reward:int, pre_terminated:bool):
        '''
        步骤: 1. 如果没有创建policy, 则创建一个 
        2. 对于当前的这个obs, 提取其符号，并判断是否为终点 or 已经出现的转移符号
        3. 正常探索
        '''
        pass
        # if not self.create_flag:
        #     self.create_policy()
        # symbol = self.symbol_manager.generate_symbol(obs)
        # if pre_reward >= 1: # 如果奖励大于等于1， 说明到达了终点。
        #     self.symbol_manager.get_anomaly_symbol()
        #     self.symbol_manager.reset()  # 重置符号管理器
        #     return -1
        # else:
        #     if self.symbol_manager.check_symbol_exist(symbol):
        #         self.symbol_manager.get_anomaly_symbol()
        #         self.symbol_manager.reset()  # 重置符号管理器
        #         return -1
        #     else:
        #         action = self.atomic_policy_manager.get_action(self.new_policy_index, obs)
        #         return action
            
            

        
            

