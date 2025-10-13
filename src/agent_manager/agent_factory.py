from .agents.base_agent.agent_interface import AgentInterface
import importlib

class AgentFactory:
    @staticmethod
    def create_agent(conf: dict) -> AgentInterface:
        agent_name = conf['agent']['name']
        # 动态导入模块
        module = importlib.import_module(
                f".agents.{agent_name}.agent", package=__package__
            )
        Agent = getattr(module, "Agent")
        return Agent(conf)
