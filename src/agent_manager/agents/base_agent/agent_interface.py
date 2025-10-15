from abc import abstractmethod, ABC

class AgentInterface(ABC): 
    def __init__(self):
        pass
    
    @abstractmethod
    def inference(self, idx, emv, max_steps):
        pass

    @abstractmethod
    def learn(self, idx, env, max_steps):
        pass