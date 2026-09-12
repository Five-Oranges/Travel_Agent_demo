import os
import re

from Ai_Tool import OpenAICompatibleClient
from Weather_Tool import get_weather
from Choice_Tool import get_attraction


# --- 1. 配置 LLM 客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = os.environ.get('DEEPSEEK_API_KEY')
BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-flash"


llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 定义系统 Prompt 与可用工具 ---
AGENT_SYSTEM_PROMPT = """
你是一个智能旅游助手，能够根据用户请求，通过 Thought-Action-Observation 的 ReAct 模式完成任务。

你可以使用以下工具：
1. get_weather(city: str) -> str：查询指定城市的实时天气。
2. get_attraction(city: str, weather: str) -> str：根据城市和天气推荐合适的旅游景点。

请严格遵循以下输出格式：
Thought: 你的思考过程
Action: 工具调用，例如 get_weather(city="北京")
Observation: （由系统填充工具返回结果）

当你认为已经获得足够信息可以回答用户时，请输出：
Thought: 你的思考过程
Action: Finish[最终答案]
"""

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# --- 3. 初始化 ---
user_prompt = "你好，请帮我查询一下今天金华的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]
print(f"用户输入: {user_prompt}\n" + "=" * 40)

# --- 4. 运行主循环 ---
for i in range(5):  # 设置最大循环次数
    print(f"--- 循环 {i + 1} ---\n")

    # 4.1. 构建 Prompt
    full_prompt = "\n".join(prompt_history)

    # 4.2. 调用 LLM 进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

    # 模型可能会输出多余的 Thought-Action，需要截断
    match = re.search(
        r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
        llm_output,
        re.DOTALL
    )
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")

    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 4.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue

    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误: 未定义的工具 '{tool_name}'"

    # 4.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
