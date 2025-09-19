
from loguru import logger
from .env_interface import EnvInterface

class EnvFactory:
    @staticmethod
    def create_env(conf: dict) -> EnvInterface:
        """
        通过配置创建环境
        """
        if conf["env_config"]["type"] == "minigrid":
            from .minigrid_env import FinalEnv

            return FinalEnv(conf["env_config"])

        elif conf["env_config"]["type"] == "minihack":
            from .minihack_env import FinalEnv

            return FinalEnv(conf["env_config"])
        
        else:
            logger.error(f"环境配置参数错误，目前只支持minihack和minigrid，而您的参数为{conf['env_config']['type']}")
            exit()
    
