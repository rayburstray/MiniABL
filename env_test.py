import hydra
from omegaconf import DictConfig
from loguru import logger
from miniabl.env.minigrid_env import make_minigrid_env
import matplotlib.pyplot as plt



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    # print(cfg["env"]["minigrid_maps"])
    env = make_minigrid_env(
        env_key="1",
        maps=cfg["env"]["minigrid_maps"],
        parallel=cfg["env"]["parallel"],
        havekey=True
    )
    obss, _ = env.reset()
    print(obss["image"].shape)
    img1 = obss["image"][0]
    plt.imshow(img1)
    plt.show()

    
if __name__ == "__main__":
    main()