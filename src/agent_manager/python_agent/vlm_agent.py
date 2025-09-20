from .python_agent_interface import PythonAgent
from loguru import logger
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from volcenginesdkarkruntime import Ark
import json

def encode_image(img_array: np.ndarray) -> str:
    """
    用于将numpy矩阵图片 编码成ase64编码，以便使用vlm模型进行识别
    """

    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # 编码为base64
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return base64_str


'''
注意：vlm的act的输入obs应该是原始图片，而不是经过视觉识别后提取的符号矩阵。
因此，需要确保在使用vlm的时候，视觉识别模块应该设置为none，即不经过处理，而不是什么例如cnn之类的东西。
'''

def act(action: str) -> int:
    if action == "north":
        return 0
    elif action == "east":
        return 1
    elif action == "south":
        return 2
    elif action == "west":
        return 3
    elif action == "pickup":
        return 8
    elif action == "apply":
        return 9
    elif action == "open":
        return 10
    else:
        return -1

class MyAgent(PythonAgent):
    def __init__(self, conf: dict):
        super().__init__(conf)
        self.api_key = conf['vlm_config']['api_key']
        self.model_name = conf['vlm_config']['model_name']
        self.prompt = conf['vlm_config']['prompt']  
        self.tools = [conf['vlm_config']['tools']]
        self.volc_client = Ark(api_key = self.api_key)
        self.history = []
        

    

    def act(self, obs: np.ndarray[int, np.dtype[np.int_]]):
        img_base64 = encode_image(obs)
        self.history.append(
            {
                "role": "system",
                "content": self.prompt
            }
        )
        self.history.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/png;base64,{img_base64}",
                        "detail": "high"
                    }
                }
            ]
        })
        # messages = [
        #     {
        #         "role": "system",
        #         "content": self.prompt
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url":  f"data:image/png;base64,{img_base64}",
        #                     "detail": "high"
        #                 }
        #             }
        #         ]
        #     }
        # ]
        try:
            completion = self.volc_client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                tools=self.tools
            )
            logger.info(f"vlm_agent: {completion.choices[0]}")
            tool_call = completion.choices[0].message.tool_calls[0]
            function_args_str = tool_call.function.arguments  # 获取JSON字符串
            # 解析JSON字符串为字典
            function_args = json.loads(function_args_str)

            # 提取具体参数（例如提取"action"字段）
            action = function_args.get("action")  # 结果为"north"之类的字符串
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"执行动作{action}"
                }
            )
            return act(action)
        except Exception as e:
            logger.error(f"vlm_agent: {e}")
        

