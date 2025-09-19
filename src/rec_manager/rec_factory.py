from .rec_interface import RecInterface
from loguru import logger

class RecFactory:
    @staticmethod
    def create_rec(conf: dict) -> RecInterface:
        if conf["vision_config"]["type"] == "none":
            from .none_rec import VisualRec

            return VisualRec(conf["vision_config"])
        elif conf["vision_config"]["type"] == "cnn":
            from .cnn_rec.cnn_rec import VisualRec

            return VisualRec(conf["vision_config"]["cnn_config"])
        
        else:
            logger.error(f"视觉识别模块参数错误，目前只支持cnn与none，而您的参数为{conf['vision_config']['type']}")
            exit()
