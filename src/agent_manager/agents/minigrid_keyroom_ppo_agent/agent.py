import re
from ...agents.base_agent.agent_interface import AgentInterface
from ...agents.minigrid_keyroom_ppo_agent.submodules.action_model import ActionModel
from ...agents.minigrid_keyroom_ppo_agent.submodules.memory import Memory
import numpy as np
import os
from tqdm import tqdm
from loguru import logger
from PIL import Image

class Agent(AgentInterface):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = conf
        self.action_model = ActionModel(conf)
        self.memory = Memory(conf)
        self.connect()
        pass

    def inference(self, idx, env, max_steps):
        obs, info = env.reset()
        num_envs = env.num_envs
        reward = np.zeros(num_envs)
        max_steps = int(float(max_steps))
        terminated, truncated = np.zeros(num_envs), np.zeros(num_envs)
        #logger.info((obs, reward.shape, terminated.shape, truncated.shape))
        #exit()
        for i in tqdm(range(max_steps)):
            actions = self.action_model.act(obs['image'])
            obs, reward, terminated, truncated, info = env.step(actions)
            if np.any( reward>0 ):
                logger.info('zhongda shushu')
            self.save_obs( idx, i, obs['image'] )
    def connect(self):
        self.action_model.memory = self.memory

    def learn(self, idx, env, max_steps):
        obs, info = env.reset()
        obs =obs['image']
        num_envs = env.num_envs
        reward = np.zeros(num_envs)
        max_steps = int(float(max_steps))
        self.memory.update(0, 'max_steps', max_steps)
        steps_per_train = self.conf['envs']['steps_per_train']
        
        done = None
        for i in tqdm(range(max_steps)):
            self.memory.update(0, 'iteration', i)
            # action, logprob, value = self.action_model.learn(obs['image'], reward, terminated, truncated, info)
            obs, done, env = self.action_model.learn(obs, done, env)
            if i % steps_per_train == 0 and i > 0:
                self.action_model.update( obs, done )

        pass

    def save_obs(self, env_idx, obs_idx, obs):
        save_folder_path = f'tmp_data/imgs/env{env_idx}'
        parallel = self.conf['envs']['parallel']
        for i in range(parallel):
            sub_save_folder_path = os.path.join(save_folder_path, f'sub_env{i}')
            if not os.path.exists(sub_save_folder_path):
                os.mkdir(sub_save_folder_path)
            img = Image.fromarray(obs[i])
            img.save(os.path.join(sub_save_folder_path, f'{obs_idx}.png'))