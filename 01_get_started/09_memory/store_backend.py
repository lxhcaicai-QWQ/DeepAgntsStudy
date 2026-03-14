import os
from urllib.request import urlopen

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from deepagents.backends.utils import create_file_data
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

#==================================

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

#==================================

with urlopen("https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md") as response:
    agents_md = response.read().decode("utf-8")

# Create the store and add the file to it
store = InMemoryStore()
file_data = create_file_data(agents_md)
store.put(
    namespace=("filesystem",),
    key="/AGENTS.md",
    value=file_data
)

agent = create_deep_agent(
    backend=(lambda rt: StoreBackend(rt)),
    store=store,
    memory=[
        "/AGENTS.md"
    ]
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Please tell me what's in your memory files.",
            }
        ],
        "files": {"/AGENTS.md": create_file_data(agents_md)},
    },
    config={"configurable": {"thread_id": "12345"}},
)

print("result: "+ result["messages"][-1].content)