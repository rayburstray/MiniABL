from ..rec_interface import RecInterface
from .cnn import load_model
from .dataset_build import MinihackDataset
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor  # 导入 ToTensor

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

tile_size = (16, 16)


class VisualRec(RecInterface):
    def __init__(self, conf: dict):
        self.conf = conf
        self.model = load_model(self.conf["num_classes"])
        self.to_tensor = ToTensor()

    def recognize(self, obs: np.ndarray) -> np.ndarray:
        print(obs.shape)
        MinihackDataset.update(img=obs)
        v, h, _ = obs.shape
        v_times, h_times = v // tile_size[0], h // tile_size[1]
        rec = []
        for i in range(v_times):
            rec_list = []
            for j in range(h_times):
                tile = obs[
                    i * tile_size[0] : (i + 1) * tile_size[0],
                    j * tile_size[1] : (j + 1) * tile_size[1],
                ]
                # 在使用tile输入模型前进行转换
                tile = Image.fromarray(tile)
                tile_tensor = self.to_tensor(
                    tile
                )  # 转换为Tensor（注意可能需要调整数据类型）
                tile_tensor = tile_tensor.unsqueeze(0).to(device)
                tile_tensor = tile_tensor.to(device)  # 转移到模型所在设备
                tile_rec = self.model(tile_tensor)
                tile_rec = tile_rec.squeeze(0)
                tile_rec = tile_rec.argmax(dim=0)
                rec_list.append(tile_rec.cpu().detach().numpy())
            rec.append(rec_list)
        return np.array(rec)
