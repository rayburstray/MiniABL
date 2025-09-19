from .python_agent_interface import PythonAgent
import numpy as np
from collections import deque
from enum import Enum
from loguru import logger
from typing import Dict, Tuple, Optional  # 新增：用于显式声明类型

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


class MyAgent(PythonAgent):
    def __init__(self, conf: dict):
        super().__init__(conf)
        self.act_queue = deque()  # 决策队列
        self.see_key = False  # 是否看见钥匙
        self.has_key = False  # 是否持有钥匙
        self.see_door = False  # 是否看见门
        self.open_door = False  # 是否打开门
        self.see_final = False  # 是否看见最终点
        self.constant = False  # 是否正在执行不可打断行为

    def act(self, obs: np.ndarray[int, np.dtype[np.int_]]):
        logger.info(obs[7:17, 35:55])
        self.update_state(obs)  # 更新观测状态
        if self.constant and self.act_queue:
            return self.act_queue.popleft()
        if not self.act_queue:
            self.constant = False  # 解除不可打断行为
        if not self.see_key:
            self.bfs(obs, Object.EMPTY.value)
            return self.act_queue.popleft()
        if not self.has_key:
            self.bfs(obs, Object.KEY.value)
            self.act_queue.append(Action.Pickup.value)  # 获取钥匙
            self.has_key = True
            self.constant = True
            return self.act_queue.popleft()
        if not self.see_door:
            self.bfs(obs, Object.EMPTY.value)
            return self.act_queue.popleft()
        if not self.open_door:
            if self.boy_next_door(obs):  # 检测是否相邻门
                self.act_queue.clear()
                self.act_queue.append(Action.Apply.value)
                self.act_queue.append(Action.Open.value)
                self.open_door = True
                self.constant = True
                return self.act_queue.popleft()
            self.bfs(obs, Object.DOOR.value)
            return self.act_queue.popleft()
        if not self.see_final:
            self.bfs(obs, Object.EMPTY.value)
            return self.act_queue.popleft()
        else:
            self.bfs(obs, Object.FINAL.value)
            self.constant = True
            return self.act_queue.popleft()

    def update_state(self, obs: np.ndarray[int, np.dtype[np.int_]]):
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i][j] == Object.KEY.value:
                    self.see_key = True
                elif obs[i][j] == Object.DOOR.value:
                    self.see_door = True
                elif obs[i][j] == Object.FINAL.value:
                    self.see_final = True

    def locate(self, obs: np.ndarray[int, np.dtype[np.int_]], index: int) -> tuple:  # 定位特定物体捏
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i][j] == index:
                    return i, j
        logger.info(f'未找到对象{index}')
        return -1, -1

    def bfs(self, obs: np.ndarray[int, np.dtype[np.int_]], target: int):
        """
        BFS算法，从start开始搜索，直到找到目标对象target
        """
        start = self.locate(obs, Object.PLAYER.value)
        queue = deque([start])
        visited = np.zeros_like(obs, dtype=bool)
        visited[start[0]][start[1]] = True

        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while queue:
            cur = queue.popleft()

            if obs[cur[0]][cur[1]] == target:
                path = deque()
                node = cur
                while node is not None:
                    path.appendleft(node)
                    node = parent[node]
                actions = self.transform_from_pos_to_action(path)

                self.act_queue.clear()
                self.act_queue.extend(actions)
                return

            for dx, dy in directions:
                next_x, next_y = cur[0] + dx, cur[1] + dy

                # 检查边界
                if 0 <= next_x < obs.shape[0] and 0 <= next_y < obs.shape[1]:
                    # 检查是否可移动且未访问过
                    next_type:int = int(obs[next_x][next_y])
                    if not visited[next_x][next_y] and self.move_available(
                        next_type
                    ):
                        visited[next_x][next_y] = True
                        queue.append((next_x, next_y))
                        parent[(next_x, next_y)] = cur

    def move_available(self, type: int) -> bool:
        if type == Object.WALL.value:
            return False
        return True

    def transform_from_pos_to_action(self, pos_deque: deque) -> deque:
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

    def boy_next_door(self, obs: np.ndarray[int, np.dtype[np.int_]]) -> bool:
        start = self.locate(obs, Object.PLAYER.value)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dx, dy in directions:
            next_x, next_y = start[0] + dx, start[1] + dy
            if 0 <= next_x < obs.shape[0] and 0 <= next_y < obs.shape[1]:
                if obs[next_x][next_y] == Object.DOOR.value:
                    return True
        return False
