from abc import ABC, abstractmethod

class MemoryInterface(ABC):
    def __init__(self, conf):
        self.conf = conf
        pass

    @abstractmethod
    def update(self, memory_key, new_memory_value):
        pass

    @abstractmethod
    def get_memory(self, memory_key):
        pass