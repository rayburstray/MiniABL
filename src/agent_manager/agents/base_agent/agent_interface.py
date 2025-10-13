from abc import abstractmethod, ABC

class AgentInterface(ABC): 
    def __init__(self):
        pass
    
    @abstractmethod
    def learn(self, env):
        pass

    @abstractmethod
    def inference(self, env):
        pass