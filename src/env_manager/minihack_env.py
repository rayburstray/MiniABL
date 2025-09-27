from .env_interface import EnvInterface
import gymnasium as gym
import numpy as np
from nle import nethack
from loguru import logger
import minihack


# minihack 可选择的环境列表：https://minihack.readthedocs.io/en/latest/envs/index.html
class FinalEnv(EnvInterface):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.task = conf["task"]
        MOVE_ACTIONS = tuple(nethack.CompassDirection)
        ALL_ACTIONS = MOVE_ACTIONS + (
            nethack.Command.PICKUP,
            nethack.Command.APPLY,  # 使用钥匙（应用工具）
            nethack.Command.OPEN,  # 开门（适用于已解锁的门）)
        )
        logger.info(MOVE_ACTIONS)
        self.env = gym.make(
            self.task,
            observation_keys=("glyphs", "chars", "colors", "pixel"),
            actions=ALL_ACTIONS,
        )
        self.obs, _ = self.env.reset()
        self.save_img(self.obs["pixel"])
        # logger.info(self.obs['glyphs'].shape)

    def render(self) -> np.ndarray:
        # logger.info('获取环境图像')
        return self.obs["pixel"]

    def step(self, action:int) -> tuple[np.ndarray, int, bool]:
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        self.save_img(self.obs["pixel"])
        self.save_policy(action)
        return (self.obs, reward, terminated or truncated)
