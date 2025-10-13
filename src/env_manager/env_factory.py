
from loguru import logger
from .env_interface import EnvsInterface
import minihack

class EnvsFactory:
    @staticmethod
    def create_envs(conf: dict) -> EnvsInterface:
        """
        通过配置创建任务集
        """
        if conf['envs']['base'] == 'minigrid':
            from.minigrid_env import FinalEnvs
            return FinalEnvs(conf)
            pass
        elif conf['envs']['base'] == 'minihack':
            from.minihack_env import FinalEnvs
            return FinalEnvs(conf)
            pass