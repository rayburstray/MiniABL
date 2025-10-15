from .env_interface import  EnvsInterface
import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper
from loguru import logger
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

'''
0: left 0: 向左

1: right 1: 右

2: up 2: 上

3: toggle 3: 切换

4: pickup 4: 拾取

5: drop 5: 丢弃

6: done/noop 6: 完成/无操作
'''

    

class CustomMinigridEnv(MiniGridEnv):
    def __init__(
            self,
            size = 8,
            width = None,
            height = None,
            max_steps: int | None = None, # 这里的max_steps是max_steps_per_episode, 而非max_steps_per_map
            map: list[list] | None = None,
            **kwargs,
        ):

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.map = map

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space = mission_space,
            width = width,
            height = height,
            see_through_walls = True,
            max_steps = max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "miniABL's minigrid custom mission."
    
    
    def handle_map(self, width, height, map_char):
        if map_char == '-':
            return
        elif map_char == 'S':
            self.agent_pos = (width , height)
        elif map_char == 'G':
            self.put_obj(Goal(), width, height)
        elif map_char == 'x':
            self.put_obj(Wall(), width, height)
        elif map_char == 'E':
            self.put_obj(Lava(), width, height)
        elif map_char == 'D':
            self.put_obj(Door(COLOR_NAMES[0], is_locked=True), width, height)
        elif map_char == 'K':
            self.put_obj(Key(COLOR_NAMES[0]), width, height)
        else:
            logger.error("Unknown map character: %s" % map_char)
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.agent_dir = 0 # agent的朝向，不知道有啥用捏
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                self.handle_map(i, j, self.map[j][i])
                


class FinalEnvs(EnvsInterface):
    def __init__(self, conf: dict):
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
                env = gym.make(env_name, highlight=False)
                pass
            elif env_to_build_config['type'] == 'custom':
                env_map = np.array(env_to_build_config['map'])
                height = env_map.shape[0]
                width = env_map.shape[1]
                env = CustomMinigridEnv(
                    width = width,
                    height = height,
                    max_steps = env_to_build_config['max_steps_per_episode'],
                    map = env_map,
                    highlight = False
                )

            env = RGBImgObsWrapper(env)
            return env
        return thunk
    

    def get_current_env(self):
        return self.envs[self.current_env_idx]
    
    def update_env(self):
        self.current_env_idx += 1
        if self.current_env_idx >= self.env_nums:
            logger.info('所有任务已经跑完，结束程序捏')
            exit()
