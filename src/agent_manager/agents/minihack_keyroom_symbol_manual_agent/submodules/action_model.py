from ...base_agent.submodules.action_model import ActionModelInterface
from loguru import logger


class ActionModel(ActionModelInterface):
    def __init__(self, conf: dict):
        super().__init__(conf)


    def act(self, obs_processed):
        if self.memory is None:
            logger.error('memory is None')
            exit()
        return self.kb.query('act', obs_processed, self.memory)

       

