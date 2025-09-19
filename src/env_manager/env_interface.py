import numpy as np
from abc import abstractmethod, ABC
import os
from loguru import logger
from PIL import Image

tmp_save_path = "tmp_data"


class EnvInterface(ABC):
    def __init__(self):
        """
        初始化
        """
        self.init_tmp_folder()

        pass

    @abstractmethod
    def render(self) -> np.ndarray:  # 返回图像
        pass

    @abstractmethod
    def step(self, action:int) -> tuple[np.ndarray, bool]:  # 交互
        """
        交互，返回当前状态和是否结束。

        Returns:
            tuple[np.ndarray, bool]: 当前观测和是否结束。
        """
        pass

    def init_tmp_folder(self):
        if not os.path.exists(tmp_save_path):
            logger.info("检测到没有临时文件夹，正在创建临时文件夹...")
            os.mkdir(tmp_save_path)
        img_tmp_path = os.path.join(tmp_save_path, "imgs")
        policy_tmp_path = os.path.join(tmp_save_path, "policy.txt")
        # 图片临时文件夹
        if os.path.exists(img_tmp_path):
            logger.info("正在删除原有图片临时文件夹里面的内容...")
            for file in os.listdir(img_tmp_path):
                os.remove(os.path.join(img_tmp_path, file))
        else:
            os.mkdir(img_tmp_path)

        # 策略临时文件
        if os.path.exists(policy_tmp_path):
            logger.info("正在删除原有策略临时文件...")
            os.remove(policy_tmp_path)
        with open(policy_tmp_path, "w", encoding="utf-8") as f:
            # 可以向文件中写入内容，若仅需创建空文件，此句可省略
            f.write("这是的policy文件")
        logger.info("临时文件夹初始化完毕")

    def save_img(self, img):
        img_tmp_path = os.path.join(tmp_save_path, "imgs")
        if not os.path.exists(img_tmp_path):
            logger.error(f"临时文件夹{img_tmp_path}不存在，请检查！")
            return
        img_index = len(os.listdir(img_tmp_path))
        img_path = os.path.join(img_tmp_path, f"{img_index}.png")
        img = Image.fromarray(img)
        img.save(img_path)
        logger.info(f"图片已保存至: {img_path}")

    def save_policy(self, action):
        policy_tmp_path = os.path.join(tmp_save_path, "policy.txt")
        with open(policy_tmp_path, "a") as f:
            f.write(str(action))

    def generate_gif(self):
        img_tmp_path = os.path.join(tmp_save_path, "imgs")
        gif_path = os.path.join(tmp_save_path, "animation.gif")

        img_files = [f for f in os.listdir(img_tmp_path) if f.endswith(".png")]
        img_files.sort(key=lambda x: int(x.split(".")[0]))  # 按文件名中的数字排序
        # 读取所有图片
        images = []
        for img_file in img_files:
            img_path = os.path.join(img_tmp_path, img_file)
            img = Image.open(img_path)
            images.append(img)
        # 保存为GIF动画
        if len(images) > 1:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=500,  # 每帧持续时间（毫秒）
                loop=0,  # 无限循环
            )
        logger.info(f"GIF动画已保存至: {gif_path}")
