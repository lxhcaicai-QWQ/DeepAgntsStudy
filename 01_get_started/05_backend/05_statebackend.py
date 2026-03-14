# Under the hood, it looks like
import os

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_openai import ChatOpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL")      # 例如 https://api.your-service.com/v1
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL")

llm = ChatOpenAI(
    model=MODEL_NAME,           # 模型名，会透传给服务端
    base_url=BASE_URL,          # 自定义 base_url
    api_key=API_KEY,            # 你的 API key
    temperature=0.7
    # max_tokens=..., timeout=..., 其他参数也可以直接写
)

agent = create_deep_agent(
    model=llm,
    backend=(lambda rt: StateBackend(rt))   # Note that the tools access State through the runtime.state
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "你好，我叫caicai"}]})
result2 = agent.invoke({"messages": [{"role": "user", "content": "你还记得我名字吗？"}]})

print("result1: "+ result1["messages"][-1].content)
print("result2: "+ result2["messages"][-1].content)