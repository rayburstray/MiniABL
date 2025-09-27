import os
from loguru import logger
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest


transition_symbol_storage_path = 'src/agent_manager/python_agent/a2rl/symbol_storage'

class SymbolManager:
    def __init__(self):
        if not os.path.exists(transition_symbol_storage_path):
            os.mkdir(transition_symbol_storage_path)
            logger.info('检测到未创建符号库，已为您创建')

        self.last_image = None # 上一张图片
        self.symbol_nums = 0 # 符号数量
        self.symbol_trajectory = [] 

    def reset(self):
        self.last_image = None
        self.symbol_trajectory = []

    def generate_symbol(self, image): # 利用图片生成符号，并自动保存至trajactory中
        if self.last_image is None:
            logger.info('符号管理器：由于无前置图片，所以此次不生成符号')
            return None
        # 计算图片差值
        diff = np.abs(image - self.last_image)
        # 提取边缘
        non_zero_coords = np.where(diff > 0.05)
        # 裁剪出可以框住所有非0值的最小矩阵


        # 计算最小包围矩阵的边界
        min_row, max_row = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
        min_col, max_col = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])

        # 裁剪出最小矩阵（包含所有非0值）
        min_matrix = diff[min_row:max_row+1, min_col:max_col+1]

        symbol_image = Image.fromarray(min_matrix)

        self.last_image = image
        self.symbol_trajectory.append(symbol_image)
        return symbol_image

    
    def check_symbol_exist(self, symbol) -> bool: # 用于检测存储库中是否已经存在了该符号 True: 存在 False: 不存在
        for symbol_name in os.listdir(transition_symbol_storage_path):
            if symbol_name.endswith('.png'):
                symbol_image = Image.open(os.path.join(transition_symbol_storage_path, symbol_name))
                symbol_np = np.array(symbol_image)
                if symbol_np == symbol:
                    return True
        return False
    
    def get_symbol(self, index) -> np.ndarray: # 获取指定索引的符号
        symbol_image = Image.open(os.path.join(transition_symbol_storage_path, f'{index}.png'))
        symbol_np = np.array(symbol_image)
        return symbol_np
    
    def get_anomaly_symbol(self): # 利用隔离森林从trajectory获取异常符号
        
