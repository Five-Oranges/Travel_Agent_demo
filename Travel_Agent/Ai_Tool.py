# # Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

# client = OpenAI(
#     api_key=os.environ.get('DEEPSEEK_API_KEY'),
#     base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-flash",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False,
#     reasoning_effort="high",
#     extra_body={"thinking": {"type": "enabled"}}
# )

# print(response.choices[0].message.content)


class OpenAICompatibleClient:
    """一个用于调用任何兼容 OpenAI 接口的 LLM 服务的客户端。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM API 来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用 LLM API 时发生错误: {e}")
            return "错误:调用 LLM API 服务时出错。"
