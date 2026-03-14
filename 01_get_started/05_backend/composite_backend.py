import os

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore


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


composite_backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={
        "/memories/": StoreBackend(rt),
    }
)

agent = create_deep_agent(
    model=llm,
    backend=composite_backend,
    store=InMemoryStore()  # Store passed to create_deep_agent, not backend
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "帮我创建个test.txt,放在memories文件夹下面,记录我今天出深圳湾公园玩的计划"}]})
result2 = agent.invoke({"messages": [{"role": "user", "content": "从test.txt获取的内容是什么"}]})

print("result1: "+ result1["messages"][-1].content)
print("result2: "+ result2["messages"][-1].content)