from .atomic_policy_manager import AtomicPolicyManager
from .symbol_manager import SymbolManager
from ..a2rl_agent import MyAgent, Behavior
import numpy as np
import random

# class DDM:
#     def __init__(self, symbol_manager:SymbolManager):
#         self.symbol_manager = symbol_manager
#         self.evidences = [0] + [0 for i in range(symbol_manager.symbol_nums)]

#         pass

#     def make_decision(self):
#         fail_counts = 0
#         for i in range(len(self.evidences)):
#             if self.evidences[i] > 100:
#                 return i # 将其接入i号ppo agent，转向train阶段
#             if self.evidences[i] < -100:
#                 fail_counts += 1
#         if fail_counts == len(self.evidences):
#             return -1 # 转向discover阶段
#         return None # 继续test
            
       
        
        

class A2RLTest:
    def __init__(self, conf:dict, a2rl_agent:MyAgent, atomic_policy_manager:AtomicPolicyManager, symbol_manager:SymbolManager):
        self.conf = conf
        self.a2rl_agent = a2rl_agent
        self.atomic_policy_manager = atomic_policy_manager
        self.symbol_manager = symbol_manager
        self.ddm = DDM(self.symbol_manager)
        self.current_policy = -1
        self.policy_nums = self.atomic_policy_manager.policy_nums

    # def sample_policy(self):
    #     self.current_policy = random.randint(0, self.policy_nums - 1)
    #     self.symbol_manager.reset()
    def act(self, obs: np.ndarray[int, np.dtype[np.int_]], pre_reward:int, pre_terminated:bool) -> int:
        '''
        这里的reward是env自带的reward.
        步骤: 1. 检测DDM是否应该做出判断
            2. 检测是否是异常状态，为上一步做出的规划累计DDM证据
            3. 初始化检测
            4. 进行推理并返回动作
        ''' 
        pass
        # # step 1
        # if self.ddm.make_decision() >= 0: # 训练
        #     self.a2rl_agent.changhe_state(Behavior.Train)
        # elif self.ddm.make_decision() == -1: # 发现
        #     self.a2rl_agent.changhe_state(Behavior.Discover)
        # else: # 测试
        #     # step 2
        #     symbol = self.symbol_manager.generate_symbol(obs)
        #     if self.current_policy == -1: # 如果还没有正在运行的policy，就采样一个
        #         self.sample_policy()
        #     if pre_reward >= 1:
        #         self.ddm.evidences[self.current_policy] += 50 # 到达终点
        #         self.sample_policy() # 采样一个新policy
        #     else:
        #         if symbol is not None and self.symbol_manager.check_symbol_exist(symbol):
        #             self.ddm.evidences[self.current_policy] += 50 # 走到转折点
        #         else:
        #             self.ddm.evidences[self.current_policy] -= 1 # 暂时失败

        # # step 4
        # action = self.atomic_policy_manager.get_action(self.current_policy, obs)
        # return action

        
