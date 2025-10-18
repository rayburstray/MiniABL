import random
# from gymnasium.core import ActType, ObsType
# from typing import Any, SupportsFloat
# import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from loguru import logger
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
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
        maps: list[list[list]], # 提前定义的这个环境可能的所有地图
        select_map_list: list[int], # 选中的地图的列表。如果只选中一个地图，那么使用该地图；如果选中不止一个地图，那么每次reset随机抽取一个地图进行初始化。（为了支持随机环境）
        # size: int = 8,
        # width: int = None,
        # height: int = None,
        max_steps: int | None = None, # 这里的max_steps是max_steps_per_episode, 而非max_steps_per_map
        havekey: bool = False, # whether the agent have the key
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.maps = maps
        self.select_map_list = select_map_list
        self.select_map_num = len(select_map_list)

        if max_steps is None:
            max_steps = 256 # 4 * size**2

        self.havekey = havekey
        self.width = len(maps[select_map_list[0]])
        self.height = len(maps[select_map_list[0]][0])

        super().__init__(
            mission_space = mission_space,
            width = self.width,
            height = self.height,
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
        
        if self.select_map_num == 1:
            selected_map = self.maps[self.select_map_list[0]]
            assert len(selected_map) == self.grid.width and len(selected_map[0]) == self.grid.height
            for i in range(self.grid.width):
                for j in range(1, self.grid.height):
                    self.handle_map(i, j, selected_map[j][i])
        else:
            selected_map = self.maps[random.choice(self.select_map_list)]
            assert len(selected_map) == self.grid.width and len(selected_map[0]) == self.grid.height
            for i in range(self.grid.width):
                for j in range(1, self.grid.height):
                    self.handle_map(i, j, selected_map[j][i])


    def gen_obs(self): # 每次reset和step都会调用此函数返回obs
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # Add the inventory (物品栏) && havekey
        if self.step_count == 0 and self.havekey:
            self.carrying = Key("blue")
        if self.carrying:
            print(self.carrying)
            self.grid.set(0, 0, self.carrying)

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        print(image)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs
            
                
def make_minigrid_env(
    env_key: str, # now is unnecessary, but will be used in the future
    parallel: int,
    maps: list[list[list]],
    select_map_list: list[int] = [0],
    max_steps: int = 256,
    havekey: bool = False,
    **kwargs
):
    def make_single_env():
        def thunk() -> gym.Env:
            env = CustomMinigridEnv(
                maps=maps,
                select_map_list=select_map_list,
                max_steps=max_steps,
                havekey=havekey,  
                highlight=False,
                **kwargs
            )
            env = RGBImgObsWrapper(env)
            return env
        return thunk

    parallel_env = gym.vector.SyncVectorEnv(
        [make_single_env() for _ in range(parallel)]
    )
    
    return parallel_env

