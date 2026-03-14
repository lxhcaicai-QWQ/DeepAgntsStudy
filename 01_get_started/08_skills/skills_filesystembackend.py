import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend


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

# Checkpointer is REQUIRED for human-in-the-loop
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(root_dir="."),
    skills=["./skills/"],
    interrupt_on={
        "write_file": True,  # Default: approve, edit, reject
        "read_file": False,  # No interrupts needed
        "edit_file": True    # Default: approve, edit, reject
    },
    checkpointer=checkpointer,  # Required!
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is langgraph?",
            }
        ]
    },
    config={"configurable": {"thread_id": "12345"}},
)

print("result: "+ result["messages"][-1].content)