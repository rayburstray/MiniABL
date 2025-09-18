from ..agent_interface import AgentInterface
import numpy as np
from pyswip import Prolog
import os


class PrologAgent(AgentInterface):
    def __init__(self, conf: dict):
        super().__init__()
        self.conf = conf
        self.prolog = Prolog()
        self.prolog_path = os.path.join(
            "src/agent_manager/prolog_agent", conf["name"] + ".pl"
        )
        self.prolog.consult(self.prolog_path)

    def act(self, obs: np.ndarray):
        obs_list = obs.tolist()  # prolog只能处理list，好弱
        return next(self.prolog.query(f"process_observation({obs_list}, Action)"))[
            "Action"
        ]
