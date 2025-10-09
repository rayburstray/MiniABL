from pygments import highlight
from .env_interface import EnvInterface
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


class FinalEnv(EnvInterface):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.task = conf["task"]
        self.highlight:bool = conf["minigrid_config"]["highlight"]
        if not self.task ==  "custom":
            self.env = gym.make(self.task, highlight = self.highlight)
            self.env = RGBImgObsWrapper(self.env)
        else:
            self.custom_conf = self.conf['minigrid_config']['custom_maps']
            logger.info(self.custom_conf)
            self.env = CustomMinigridEnvManager(self.custom_conf)
            

        self.obs, _ = self.env.reset()
        logger.info(self.obs.shape)
        self.save_img(self.obs)
        exit()


    def render(self):
        return self.obs
    
    def step(self, action:int)->tuple[np.ndarray, bool]:
        # action = action + 1
        # return (np.array([0,0]), False)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.save_img(self.obs)
        self.save_policy(action)
        return (self.obs, reward, terminated, truncated, info)
    
class CustomMinigridEnvManager:
    def __init__(self, conf:dict):
        self.conf = conf
        self.current_index = 1 # 当前的地图序号
        self.max_index = self.conf['map_nums']
        self.map_steps = 0 # 表示在目前地图已经走了多少步，达到最大步数后，换地图
        self.max_map_steps = conf["max_steps_per_map"]
        self.max_episode_steps = conf['max_steps_per_episode']
        self.highlight = conf["highlight"]
        self.set_env(self.current_index) # 初始化地图

    def set_env(self, index:int):
        '''
        用于设置地图
        '''
        self.map_steps = 0 # 重置地图步数
        self.env_map = np.array(self.conf[f'map{index}'])
        self.height = self.env_map.shape[0]
        self.width = self.env_map.shape[1]
        self.env = CustomMinigridEnv(
            width = self.width,
            height = self.height,
            max_steps = self.max_episode_steps,
            map = self.env_map,
            highlight = self.highlight
        )
        self.env = RGBImgObsWrapper(self.env)
        
        logger.info(f'已切换至map_{index} shape: {self.env_map.shape}')


    def step(self, action:int):
        '''
        可执行自动切换地图的功能
        '''
        self.map_steps += 1
        if action is not -1:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else: # 如果action为-1，则重置地图
            obs, info = self. env.reset()
            reward = 0
            terminated = False
            truncated = False

        if self.map_steps > self.max_map_steps: # 如果某一个地图的步数超过最大步数，则进入下一个地图
            self.current_index += 1
            if self.current_index > self.max_index: # 如果所有地图已经跑完，则结束程序
                logger.info('所有地图已经跑完，结束程序捏')
                exit()
            self.set_env(self.current_index)

        return obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        return obs['image'], info

    

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
                

