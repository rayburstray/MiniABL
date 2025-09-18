from .rec_interface import RecInterface

# 不使用任何提取，直接暴力返回图像


class VisualRec(RecInterface):
    def __init__(self, conf):
        super().__init__()

    def recognize(self, obs):
        return obs
