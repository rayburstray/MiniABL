import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from loguru import logger
import os
from PIL import Image
import numpy as np

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

tile_size = (16, 16)
tile_path = "src/rec_manager/cnn_rec/tiles"
label_path = "src/rec_manager/cnn_rec/labels.txt"


class MinihackDataset(Dataset):
    def __init__(self, transform=ToTensor()):
        if not os.path.exists(tile_path):
            os.mkdir(tile_path)
            logger.info("使用cnn识别模块，检测到未创建tiles文件夹，已为您创建")
        if not os.path.exists(label_path):
            with open(label_path, "w", encoding="utf-8"):
                logger.info("使用cnn识别模块，检测到未创建labels.txt文件，已为您创建")
        self.transform = transform
        # 提取文件名中的数字并按从小到大排序
        self.data_path = [
            os.path.join(tile_path, f)
            for f in sorted(
                os.listdir(tile_path), key=lambda x: int(os.path.splitext(x)[0])
            )
        ]
        self.data = []
        with open(label_path, "r", encoding="utf-8") as s:
            self.labels = [line.strip() for line in s]
        for i, f in enumerate(self.data_path):
            img = Image.open(f)
            label = self.labels[i]
            self.data.append((img, label))
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.data[idx][1]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(int(label), dtype=torch.long)

    @staticmethod
    def update(img: np.ndarray) -> bool:
        v, h, _ = img.shape
        v_times = int(v / tile_size[0])
        h_times = int(h / tile_size[1])

        # 加载已有tile的像素数据（转为numpy数组）
        existing_tiles = []
        tile_files = os.listdir(tile_path)
        for f in tile_files:
            file_path = os.path.join(tile_path, f)
            with Image.open(file_path) as tile:
                # 转为numpy数组存储，用于内容比较
                existing_tiles.append(np.array(tile))

        update_flag = False

        for i in range(v_times):
            for j in range(h_times):
                # 提取当前tile（假设img是numpy数组格式）
                tile = img[
                    i * tile_size[0] : (i + 1) * tile_size[0],
                    j * tile_size[1] : (j + 1) * tile_size[1],
                ]
                tile_img = Image.fromarray(tile)
                tile_np = np.array(tile_img)  # 转为numpy数组

                # 检查是否与已有tile内容相同
                is_duplicate = False
                for existing in existing_tiles:
                    if np.array_equal(tile_np, existing):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    update_flag = True
                    # 保存新tile
                    save_path = os.path.join(tile_path, f"{len(existing_tiles)}.png")
                    tile_img.save(save_path)
                    existing_tiles.append(tile_np)  # 加入列表，避免后续重复判断
                    logger.info(
                        f"已保存{len(existing_tiles)}张图片, 请及时更新{label_path}文件"
                    )

        return update_flag
