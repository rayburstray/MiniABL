import yaml
from src.agent_manager.agent_factory import AgentFactory
from src.env_manager.env_factory import EnvsFactory
from loguru import logger
def get_yaml_conf(path)->dict:
    return yaml.load(open(path), Loader=yaml.FullLoader)

if __name__ == "__main__":
    conf = get_yaml_conf('src/config_manager/new_conf_minigrid.yaml')
    logger.info(conf)
    agent = AgentFactory.create_agent(conf)
    envs = EnvsFactory.create_envs(conf)
    for i in range(conf['envs']['env_nums']):
        agent.learn(i, envs.envs[i], envs.envs_steps[i])
    




