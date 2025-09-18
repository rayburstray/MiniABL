from .agent_interface import AgentInterface
import importlib


class AgentFactory:
    @staticmethod
    def create_agent(conf: dict) -> AgentInterface:
        if conf["agent_config"]["type"] == "python":
            name = conf["agent_config"]["name"]
            # 动态导入模块
            module = importlib.import_module(
                f".python_agent.{name}", package=__package__
            )
            # 获取模块中的类
            MyAgent = getattr(module, "MyAgent")
            return MyAgent(conf["agent_config"])

        elif conf["agent_config"]["type"] == "prolog":
            name = conf["agent_config"]["name"]
            from .prolog_agent.prolog_agent_interface import PrologAgent

            return PrologAgent(conf["agent_config"])
