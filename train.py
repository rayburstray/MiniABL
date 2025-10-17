import hydra
from omegaconf import DictConfig
from loguru import logger

from miniabl.agent_manager.agent_factory import AgentFactory
from miniabl.env_manager.env_factory import EnvsFactory

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training loop.
    """
    logger.info(cfg)
    agent = AgentFactory.create_agent(cfg)
    envs = EnvsFactory.create_envs(cfg)
    for i in range(cfg.env.env_nums):
        agent.learn(i, envs.envs[i], envs.envs_steps[i])

if __name__ == "__main__":
    main()
    




