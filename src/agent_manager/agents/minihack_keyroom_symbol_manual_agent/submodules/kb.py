from ast import Tuple
from collections import deque
from zmq import has
from ...base_agent.submodules.kb import KBInterface
from typing import Any, Optional
from loguru import logger
from enum import Enum
from collections import deque
import numpy as np
from typing import Tuple, Optional, Dict

# 0-虚空，1-墙壁，2-地板，3-门，4-钥匙，5-玩家
class Object(Enum):
    EMPTY = 0
    WALL = 1
    FLOOR = 2
    DOOR = 3
    KEY = 4
    PLAYER = 5
    FINAL = 6


class Action(
    Enum
):  # 这部分在环境创建那里可以看，为了简单我没有使用斜着走的策略，所以会有跳数字的情况
    North = 0
    East = 1
    South = 2
    West = 3
    Pickup = 8
    Apply = 9
    Open = 10


class KB(KBInterface):
    def __init__(self, conf:dict):
        super().__init__(conf)
        self.rules = {}
        self.add_rule("view_obs", view_obs)
        self.add_rule("locate", locate)
        self.add_rule("move_available", move_available)
        self.add_rule("bfs", bfs)
        self.add_rule('act', act)
        pass

    def add_rule(self, rule_name: str, rule_func: Any) -> None:
        # 校验：确保传入的 rule_func 是可调用对象（避免误传非方法值）
        if callable(rule_func):
            self.rules[rule_name] = rule_func
            logger.info(f"规则 {rule_name} 已添加")
        else:
            raise ValueError(f"{rule_func} 不是可调用对象（无法作为规则添加）")

    def query(self, rule_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        # 1. 检查目标方法是否已通过 add_rule 添加
        if rule_name not in self.rules:
            logger.error(f"规则 {rule_name} 未添加，无法调用")
            return None  # 返回 None 表示方法不存在

        # 2. 取出目标方法，并用可变参数调用（兼容任意参数数目）
        target_rule = self.rules[rule_name]
        result = target_rule(*args, **kwargs)
        return result

        

def view_obs(obs_processed) -> Tuple[bool, bool, bool]:
    see_key = False 
    see_door = False
    see_final = False
    for i in range(obs_processed.shape[0]):
        for j in range(obs_processed.shape[1]):
            if obs_processed[i][j] == Object.KEY.value:
                see_key = True
            elif obs_processed[i][j] == Object.DOOR.value:
                see_door = True
            elif obs_processed[i][j] == Object.FINAL.value:
                see_final = True
    return see_key, see_door, see_final

def locate( obs_processed, index):
    for i in range(obs_processed.shape[0]):
        for j in range(obs_processed.shape[1]):
            if obs_processed[i][j] == index:
                return i, j
    return None


def move_available(idx, memory, type: int) -> bool:
        see_key, has_key, see_door, open_door, see_final = get_state(idx, memory)
        if not see_key and type == Object.DOOR.value:
            return False
        if not has_key and type == Object.DOOR.value:
            return False
        if type == Object.WALL.value:
            return False
        return True

def transform_from_pos_to_action(pos_deque: deque) -> deque:
        # 把位置队列转换成动作队列
        action_deque = deque()
        for i in range(len(pos_deque) - 1):
            if pos_deque[i][0] > pos_deque[i + 1][0]:
                action_deque.append(Action.North.value)
            elif pos_deque[i][0] < pos_deque[i + 1][0]:
                action_deque.append(Action.South.value)
            elif pos_deque[i][1] > pos_deque[i + 1][1]:
                action_deque.append(Action.West.value)
            elif pos_deque[i][1] < pos_deque[i + 1][1]:
                action_deque.append(Action.East.value)
        return action_deque


def bfs(idx, memory, obs_processed, target: int):

    start = locate(obs_processed, Object.PLAYER.value)
    if not start:
        memory.update(idx, 'see_key', False)
        memory.update(idx, 'has_key', False)
        memory.update(idx, 'see_door', False)
        memory.update(idx, 'open_door', False)
        memory.update(idx, 'see_final', False)
        memory.update(idx, 'constant', False)
        memory.update(idx, 'act_queue', deque())
        memory.update(idx, 'obs_list', [])
        return deque([0])
    queue = deque([start])
    visited = np.zeros_like(obs_processed, dtype=bool)
    visited[start[0]][start[1]] = True

    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]]= {start: None}
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:

        cur = queue.popleft()

        if obs_processed[cur[0]][cur[1]] == target:
            path = deque()
            node = cur

            while node is not None:
                path.appendleft(node)
                node = parent[node]
            actions = transform_from_pos_to_action(path)
            act_queue = deque()
            act_queue.extend(actions)
            return act_queue
        
        for dx, dy in directions:
            next_x, next_y = cur[0] + dx, cur[1] + dy

            # 边界检查
            if 0 <= next_x < obs_processed.shape[0] and 0 <= next_y < obs_processed.shape[1]:
                next_type = int(obs_processed[next_x][next_y])
                if not visited[next_x][next_y] and move_available(idx, memory, next_type):
                    queue.append((next_x, next_y))
                    visited[next_x][next_y] = True
                    parent[(next_x, next_y)] = cur

    return None
                
def update_state(idx, obs_processed, memory):
    current_see_key, current_see_door, current_see_final = view_obs(obs_processed)
    histroy_see_key, histroy_see_door, histroy_see_final =memory.get_memory(idx, 'see_key'), memory.get_memory(idx, 'see_door'), memory.get_memory(idx, 'see_final')
    new_see_key, new_see_door, new_see_final = current_see_key or histroy_see_key, current_see_door or histroy_see_door, current_see_final or histroy_see_final
    memory.update(idx, 'see_key', new_see_key)
    memory.update(idx, 'see_door', new_see_door)
    memory.update(idx, 'see_final', new_see_final)

def get_state(idx, memory):
    see_key = memory.get_memory(idx, 'see_key')
    has_key = memory.get_memory(idx, 'has_key')
    see_door = memory.get_memory(idx, 'see_door')
    open_door = memory.get_memory(idx, 'open_door')
    see_final = memory.get_memory(idx, 'see_final')
    return see_key, has_key, see_door, open_door, see_final

def boy_next_door(obs_processed) -> bool:
    start = locate(obs_processed, Object.PLAYER.value)
    if start is None:
        logger.error('不是哥们，有bug')
        exit()
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for dx, dy in directions:
        next_x, next_y = start[0] + dx, start[1] + dy
        if 0 <= next_x < obs_processed.shape[0] and 0 <= next_y < obs_processed.shape[1]:
                
                if obs_processed[next_x][next_y] == Object.DOOR.value:
                    return True
        
    return False

def act(obs_processed, memory):
    obs_processed = obs_processed.cpu().numpy().astype(int)
    actions = []
    for idx, single_obs_processed in enumerate(obs_processed):
        action = single_act(idx, single_obs_processed, memory)
        actions.append(action)
    return actions

def single_act(idx, obs_processed, memory):
    update_state(idx, obs_processed, memory)
    see_key, has_key, see_door, open_door, see_final = get_state(idx, memory)
    

    act_queue = memory.get_memory(idx, 'act_queue')
    constant = memory.get_memory(idx, 'constant')

    if act_queue and constant:
        action = act_queue.popleft()
        memory.update(idx, 'act_queue', act_queue)
        return action
    if not act_queue:
        constant = False
        memory.update(idx, 'constant', False)
    if not see_key:
        act_queue = bfs(idx, memory, obs_processed, Object.EMPTY.value)
        action = act_queue.popleft()
        memory.update(idx, 'act_queue', act_queue)
        return action
    if not has_key:
        act_queue = bfs(idx, memory, obs_processed, Object.KEY.value)
        act_queue.append(Action.Pickup.value)
        action = act_queue.popleft()
        memory.update(idx, 'has_key', True)
        memory.update(idx, 'constant', True)
        memory.update(idx, 'act_queue', act_queue)
        return action
    
    if not see_door:
        act_queue = bfs(idx, memory, obs_processed, Object.EMPTY.value)
        action = act_queue.popleft()
        memory.update(idx, 'act_queue', act_queue)
        return action
    if not open_door:
        if boy_next_door(obs_processed):
            act_queue = deque()
            act_queue.append(Action.Apply.value)
            act_queue.append(Action.Open.value)
            action = act_queue.popleft()
            memory.update(idx, 'open_door', True)
            memory.update(idx, 'constant', True)
            memory.update(idx, 'act_queue', act_queue)
            return action
        act_queue = bfs(idx, memory, obs_processed, Object.DOOR.value)
        action = act_queue.popleft()
        memory.update(idx, 'act_queue', act_queue)
        #logger.info('zhongdashushu')
        #logger.info(obs_processed[8:15, 33:52])
        return action
    if not see_final:
        act_queue = bfs(idx, memory, obs_processed, Object.EMPTY.value)
        action = act_queue.popleft()
        memory.update(idx, 'act_queue', act_queue)
        return action
    else:
        act_queue = bfs(idx, memory, obs_processed, Object.FINAL.value)
        action = act_queue.popleft()
        memory.update(idx, 'act_queue', act_queue)
        memory.update(idx, 'constant', True)
        return action
    pass