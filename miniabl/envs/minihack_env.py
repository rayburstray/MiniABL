from .env_interface import  EnvsInterface
import gymnasium as gym
import numpy as np
from nle import nethack
from loguru import logger
import minihack


# minihack 可选择的环境列表：https://minihack.readthedocs.io/en/latest/envs/index.html


class FinalEnvs(EnvsInterface):
    def __init__(self, conf:dict):
        super().__init__(conf)
        self.build_envs()


    def build_envs(self):
        for i in range(self.env_nums):
            self.envs.append(gym.vector.SyncVectorEnv(
                [self.build_env(i) for j in range(self.parallel)]
            ))
        pass

    def build_env(self, index:int):
        '''
        index 表示第i个任务
        '''
        env_to_build_config = self.conf['envs']['env_list'][index]

        def thunk():
            if env_to_build_config['type'] == 'default':
                env_name = env_to_build_config['name']
                MOVE_ACTIONS = tuple(nethack.CompassDirection)
                ALL_ACTIONS = MOVE_ACTIONS + (
                    nethack.Command.PICKUP,
                    nethack.Command.APPLY,  # 使用钥匙（应用工具）
                    nethack.Command.OPEN,  # 开门（适用于已解锁的门）)
                )
                #logger.info(MOVE_ACTIONS)
                env = gym.make(
                    env_name,
                    observation_keys=("glyphs", "chars", "colors", "pixel"),
                    actions=ALL_ACTIONS,
                )
                
                pass
            elif env_to_build_config['type'] == 'custom':
                logger.info('minihack的custom环境还没来得及写捏qaq')
                exit()
                pass
            return env
        return thunk
        pass
        