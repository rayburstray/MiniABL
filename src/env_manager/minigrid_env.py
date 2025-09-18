from .env_interface import EnvInterface


class FinalEnv(EnvInterface):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
