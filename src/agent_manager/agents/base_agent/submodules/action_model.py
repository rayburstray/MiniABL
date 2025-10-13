from abc import abstractmethod, ABC

class ActionModelInterface(ABC):
    def __init__(self, conf):
        self.kb = None
        self.memory = None
        self.conf = conf
        pass


    @abstractmethod
    def act(self,processed_obs):
        pass

