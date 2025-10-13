import numpy as np
from abc import abstractmethod, ABC
import os
from loguru import logger
from PIL import Image
import shutil

tmp_save_path = "tmp_data"

    
class EnvsInterface(ABC):
    def __init__(self, conf: dict):
        self.conf = conf
        self.env_nums = conf['envs']['env_nums']
        self.parallel = conf['envs']['parallel']
        self.envs = []
        self.envs_steps = []
        self.current_env_idx = 0
        for i in range(len(conf['envs']['env_list'])):
            self.envs_steps.append(conf['envs']['env_list'][i]['max_steps_per_map'])
        

        self.init_tmp_folder()


    @abstractmethod
    def build_envs(self):

        pass

    @abstractmethod
    def build_env(self, index:int):
        pass


    def init_tmp_folder(self):
        # 临时文件夹: tmp_data
        if not os.path.exists(tmp_save_path):
            logger.info("检测到没有临时文件夹，正在创建临时文件夹...")
            os.mkdir(tmp_save_path)
        img_tmp_path = os.path.join(tmp_save_path, "imgs")
        for i in range(self.env_nums):
            env_img_tmp_path = os.path.join(img_tmp_path, f"env{i}")
            if os.path.exists(env_img_tmp_path):
                shutil.rmtree(env_img_tmp_path)
            os.mkdir(env_img_tmp_path)
            
        

