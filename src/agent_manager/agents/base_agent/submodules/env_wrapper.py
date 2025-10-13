from abc import abstractmethod, ABC
import torch


class EnvWrapperInterface(ABC):
    def __init__(self, conf: dict):
        self.memory = None
        self.kb = None
        self.conf = conf
        self.device = torch.device(conf['device'])
        pass

    @abstractmethod
    def process(self, obs, reward, terminated, truncated, info):
        pass