from abc import abstractmethod, ABC
from typing import Any, Optional

class KBInterface(ABC):
    def __init__(self, conf):
        self.action_model = None
        self.memory = None
        self.conf = conf
        pass

    @abstractmethod
    def add_rule(self, rule_name: str, rule_func: Any) -> None:
        pass

    @abstractmethod
    def query(self, rule_name: str, *args:Any, **kwargs:Any):
        pass

