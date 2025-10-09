from sympy import im
from RL.rlkit.rlkit.torch.networks import cnn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
from loguru import logger
from PIL import Image

'''
代码改自cleanrl：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
但是为了方便一点，将源代码中的多环境改成了单环境
'''

obs_example_img_path = 'tmp_data/imgs/0.png' # 通过缓存文件夹里的这个图片来判断输入尺寸，（其实感觉这个实现怪怪的，但应该能跑通）

def layer__init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNAgent(nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = conf
        # 设定输入输出维度
        self.action_dim = conf['a2rl_config']['network']['action_dim']
        if conf['a2rl_config']['network']['input_shape'] == None:
            img_width, img_height = Image.open(obs_example_img_path).size
        else:
            img_width, img_height = conf['a2rl_config']['network']['input_shape']

        # 卷积层
        self.backbone = nn.Sequential(
            layer__init(
                nn.Conv2d(3, 8, kernel_size=8, stride=1, padding=1),
            ),
            nn.ReLU(),
            layer__init(
                nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        # 自动计算卷积层输出的尺寸
        self.backbone_out_dim = self.backbone(torch.zeros(1, 3, img_height, img_width)).shape[1]


        # 价值网络
        self.critic = nn.Sequential(
            layer__init(nn.Linear(self.backbone_out_dim, 64)),
            nn.Tanh(),
            layer__init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer__init(nn.Linear(64, 1), std=1.0),
        )
        # 策略网络
        self.actor = nn.Sequential(
            layer__init(nn.Linear(self.backbone_out_dim, 64)),
            nn.Tanh(),
            layer__init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer__init(nn.Linear(64,self.action_dim ), std=0.01),
        )

    def get_value(self, x):
        cnn_out = self.backbone(x)
        return self.critic(cnn_out)
    
    def get_action_and_value(self, x, action=None):
        cnn_out = self.backbone(x)
        logits = self.actor(cnn_out)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPO:
    def __init__(self, conf: dict):
        self.conf = conf
        self.agent = CNNAgent(conf)

    def inference(self, obs: np.ndarray[int, np.dtype[np.int_]]):
        return self.agent.get_action_and_value(torch.from_numpy(obs).float())

class AtomicPolicyManager:
    def __init__(self, conf: dict):
        self.conf = conf
        self.polocies = []
        self.policy_nums = 0
        self.create_policy()
        self.current_policy = 0
        

    def create_policy(self):
        self.polocies.append(PPO(self.conf))
        self.policy_nums += 1


    def get_action(self, index, obs: np.ndarray[int, np.dtype[np.int_]]):
        return self.polocies[index].inference(obs)
    
    