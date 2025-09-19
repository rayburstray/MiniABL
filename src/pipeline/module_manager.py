from ..env_manager.env_factory import EnvFactory
from ..agent_manager.agent_factory import AgentFactory
from ..rec_manager.rec_factory import RecFactory

from ..env_manager.env_interface import EnvInterface
from ..agent_manager.agent_interface import AgentInterface
from ..rec_manager.rec_interface import RecInterface

from loguru import logger
import yaml

config_path = "src/config_manager/conf.yaml"


class ModuleManager:
    def __init__(self, conf: dict):
        # 创建3幻神
        self.env:EnvInterface = EnvFactory.create_env(conf)
        self.agent:AgentInterface = AgentFactory.create_agent(conf)
        self.rec:RecInterface = RecFactory.create_rec(conf)

    def step(self):
        # 符号提取部分
        recognision = self.rec.recognize(self.env.render())
        # agent动作选择
        action = self.agent.act(recognision)
        logger.info(action)
        # logger.info(f'动作为{action}')
        # 环境交互
        obs, terminated = self.env.step(action)
        return obs, terminated

    def calculate(self):
        terminated = False
        step = 0
        while not terminated:
            obs, terminated = self.step()
            step += 1
        logger.info(f"游戏结束，步数{step}")
        self.env.generate_gif()


def run():
    logger.info("开始加载配置文件...")
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(conf)
    logger.info("开始创建模块管理器...")
    module_manager = ModuleManager(conf)
    module_manager.calculate()
