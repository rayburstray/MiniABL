

class EnvFactory:
    @staticmethod
    def create_env(conf: dict):
        """
        通过配置创建环境
        """
        if conf["env_config"]["type"] == "minigrid":
            from .minigrid_env import FinalEnv

            return FinalEnv(conf["env_config"])

        elif conf["env_config"]["type"] == "minihack":
            from .minihack_env import FinalEnv

            return FinalEnv(conf["env_config"])

    pass
