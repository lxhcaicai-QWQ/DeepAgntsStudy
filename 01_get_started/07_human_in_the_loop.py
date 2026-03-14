import os
import uuid

from langchain.tools import tool
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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


@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem."""
    print(f"Deleted {path}")
    return f"Deleted {path}"

@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    print(f"Contents of {path}")
    return f"Contents of {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    print(f"Sent email to {to}")
    return f"Sent email to {to}"

# Checkpointer is REQUIRED for human-in-the-loop
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=llm,
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": True,  # Default: approve, edit, reject
        "read_file": False,   # No interrupts needed
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
    },
    checkpointer=checkpointer  # Required!
)

# 为每个会话创建配置，包含 thread_id
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
# Invoke the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "发送邮件给123xxx@xxx.com 主题：周末愉快！ 内容：周末愉快！"}]},
    config=config,
    version="v2",
)

# Check if execution was interrupted
if result.interrupts:
    # Extract interrupt information
    interrupt_value = result.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]

    # Create a lookup map from tool name to review config
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    # Display the pending actions to the user
    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"Tool: {action['name']}")
        print(f"Arguments: {action['args']}")
        print(f"Allowed decisions: {review_config['allowed_decisions']}")

    # Get user decisions (one per action_request, in order)
    decisions = [
        {"type": "approve"}  # User approved the deletion
    ]

    # Resume execution with decisions
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,  # Must use the same config!
        version="v2",
    )

# Process final result
print(result.value["messages"][-1].content)
