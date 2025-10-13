from ...base_agent.submodules.memory import MemoryInterface
from collections import deque

class Memory(MemoryInterface):
    def __init__(self, conf):
        super().__init__(conf)
        parallel = self.conf['envs']['parallel']
        self.memorys = [{} for _ in range(parallel)]
        for i in range(parallel):
            self.update(i, 'see_key', False)
            self.update(i, 'has_key', False)
            self.update(i, 'see_door', False)
            self.update(i, 'open_door', False)
            self.update(i, 'see_final', False)
            self.update(i, 'constant', False)
            self.update(i, 'act_queue', deque())
            self.update(i, 'obs_list', [])


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

    