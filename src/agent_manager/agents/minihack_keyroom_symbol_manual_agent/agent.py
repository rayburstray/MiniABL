from ...agents.base_agent.agent_interface import AgentInterface
from ...agents.minihack_keyroom_symbol_manual_agent.submodules.env_wrapper import EnvWrapper
from ...agents.minihack_keyroom_symbol_manual_agent.submodules.kb import KB
from ...agents.minihack_keyroom_symbol_manual_agent.submodules.memory import Memory
from ...agents.minihack_keyroom_symbol_manual_agent.submodules.action_model import ActionModel
from loguru import logger   
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

class Agent(AgentInterface):
    def __init__(self, conf: dict):
        self.conf = conf
        self.env_wrapper = EnvWrapper(conf)
        self.kb = KB(conf)
        self.memory = Memory(conf)
        self.action_model = ActionModel(conf)
        self.connect()
        pass


    def inference(self, idx, env, max_steps): 
        obs, info = env.reset()
        num_envs = env.num_envs # 并行的环境数目
        reward = np.zeros(num_envs)
        max_steps = int(float(max_steps))
        terminated, truncated = np.zeros(num_envs), np.zeros(num_envs)
        # obs_list = []
        # obs_list.append(obs)
        for i in tqdm(range(max_steps)):
            obs_processed, reward_processed, terminated_processed, truncated_processed, info_processed = self.env_wrapper.process(obs, reward, terminated, truncated, info)
            action = self.action_model.act(obs_processed)
            logger.info(action)
            obs, reward, terminated, truncated, info = env.step(action)
            self.save_obs(idx, i, obs['pixel'])
        self.save_obs(idx)
        self.memory.reset()
        

        pass

    def learn(self, env):
        logger.info('该模型没有learn模式，请重新调整config')
        exit()
        pass

    def connect(self):
        self.action_model.kb = self.kb
        self.action_model.memory = self.memory
        self.kb.memory = self.memory
        self.env_wrapper.memory = self.memory


    def save_obs(self, env_idx, obs_idx, obs):
        save_folder_path = f'tmp_data/imgs/env{env_idx}'
        parallel = self.conf['envs']['parallel']
        for i in range(parallel):
            sub_save_folder_path = os.path.join(save_folder_path, f'sub_env{i}')
            if not os.path.exists(sub_save_folder_path):
                os.mkdir(sub_save_folder_path)
            img = Image.fromarray(obs[i])
            img.save(os.path.join(sub_save_folder_path, f'{obs_idx}.png'))
        
        

       
        