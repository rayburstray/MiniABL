

class RecFactory:
    @staticmethod
    def create_rec(conf: dict):
        if conf["vision_config"]["type"] == "none":
            from .none_rec import VisualRec

            return VisualRec(conf["vision_config"])
        elif conf["vision_config"]["type"] == "cnn":
            from .cnn_rec.cnn_rec import VisualRec

            return VisualRec(conf["vision_config"]["cnn_config"])
