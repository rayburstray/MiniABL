from ...base_agent.submodules.memory import MemoryInterface
from collections import deque

class Memory(MemoryInterface):
    def __init__(self, conf):
        super().__init__(conf)
        parallel = self.conf['envs']['parallel']
        self.memorys = [{} for _ in range(parallel)]
        for i in range(parallel):
            self.update(i, 'obs', [])
            self.update(i, 'done', [])
            self.update(i, 'reward', [])
            self.update(i, 'action', [])
            self.update(i, 'logprob', [])
            self.update(i, 'value', [])


        


    def update(self, idx, memory_key, new_memory_value):
        self.memorys[idx][memory_key] = new_memory_value
        
    
    def get_memory(self, idx, memory_key):
        if memory_key in self.memorys[idx]:
            return self.memorys[idx][memory_key]
        else:
            return None
        
    def reset(self):
        parallel = self.conf['envs']['parallel']
        self.memorys = [{} for _ in range(parallel)]
        for i in range(parallel):
            self.update(i, 'obs', [])
            self.update(i, 'done', [])
            self.update(i, 'reward', [])
            self.update(i, 'action', [])
            self.update(i, 'logprob', [])
            self.update(i, 'value', [])

    
            
