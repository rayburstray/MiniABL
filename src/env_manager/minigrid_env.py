from .env_interface import EnvInterface
import numpy as np

class FinalEnv(EnvInterface):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

    def render(self):
        return super().render()
    
    def step(self, action:int)->tuple[np.ndarray, bool]:
        action = action + 1
        return (np.array([0,0]), False)