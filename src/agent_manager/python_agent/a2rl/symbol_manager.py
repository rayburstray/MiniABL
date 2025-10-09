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

    def generate_symbol(self, image) -> np.array: # 利用图片生成符号，并自动保存至trajactory中
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
        return np.array(symbol_image)

    
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
    
    def resize_symbol(self, symbol: np.ndarray) -> np.ndarray:
        symbol_pil = Image.fromarray(symbol)
        symbol_pil = symbol_pil.resize((16, 16), Image.LANCZOS)
        return np.array(symbol_pil)

    def get_anomaly_symbol(self): # 利用隔离森林从trajectory获取异常符号
        '''
        返回元组：（ 异常分数， 符号索引， 符号数据 ）
        '''
        trajectory_resize = [self.resize_symbol(symbol) for symbol in self.symbol_trajectory[:-1]] # 注意，这里去除了最后一个符号

        X = np.array([symbol.reshape(-1) for symbol in trajectory_resize])

        model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        verbose=0  # 静默模式，不输出训练日志
        )
        model.fit(X)  # 训练模型（学习正常符号的特征分布）

        # 隔离森林的决策函数返回“负异常分数”（值越小，异常程度越高）
        negative_anomaly_scores = model.decision_function(X)
        # 转换为“正异常分数”（值越大，异常程度越高，更直观）
        anomaly_scores = -negative_anomaly_scores

        # 创建列表：每个元素是(异常分数, 符号在trajectory_resize中的索引, 原始符号数据)
        symbol_with_score = [
        (anomaly_scores[i], i, trajectory_resize[i]) 
        for i in range(len(trajectory_resize))
        ]

        # 排序后第一个元素就是异常分数最高的符号
        symbol_with_score_sorted = sorted(symbol_with_score, key=lambda x: x[0], reverse=True)
        most_anomalous = symbol_with_score_sorted[0]

        logger.info(f"最异常符号：索引={most_anomalous[1]}，异常分数={most_anomalous[0]:.4f}")

        if not self.check_symbol_exist(most_anomalous[2]):
            self.symbol_nums += 1
            symbol_image = Image.fromarray(most_anomalous[2])
            symbol_image.save(os.path.join(transition_symbol_storage_path, f'{self.symbol_nums}.png'))
            logger.info(f'符号管理器：已保存符号{self.symbol_nums}')

        return most_anomalous


