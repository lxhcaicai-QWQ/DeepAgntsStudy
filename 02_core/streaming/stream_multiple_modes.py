import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

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


agent = create_deep_agent(
    model=llm,
    system_prompt=(
        "You are a project coordinator. Always delegate research tasks "
        "to your researcher subagent using the task tool. Keep your final response to one sentence."
    ),
    subagents=[
        {
            "name": "researcher",
            "description": "Researches topics thoroughly",
            "system_prompt": (
                "You are a thorough researcher. Research the given topic "
                "and provide a concise summary in 2-3 sentences."
            ),
        },
    ],
)

#==================================


# Skip internal middleware steps - only show meaningful node names
INTERESTING_NODES = {"model_request", "tools"}

last_source = ""
mid_line = False  # True when we've written tokens without a trailing newline

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Analyze the impact of remote work on team productivity"}]},
    stream_mode=["updates", "messages", "custom"],
    subgraphs=True,
    version="v2",
):
    is_subagent = any(s.startswith("tools:") for s in chunk["ns"])
    source = "subagent" if is_subagent else "main"

    if chunk["type"] == "updates":
        for node_name in chunk["data"]:
            if node_name not in INTERESTING_NODES:
                continue
            if mid_line:
                print()
                mid_line = False
            print(f"[{source}] step: {node_name}")

    elif chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if token.content:
            # Print a header when the source changes
            if source != last_source:
                if mid_line:
                    print()
                    mid_line = False
                print(f"\n[{source}] ", end="")
                last_source = source
            print(token.content, end="", flush=True)
            mid_line = True

    elif chunk["type"] == "custom":
        if mid_line:
            print()
            mid_line = False
        print(f"[{source}] custom event:", chunk["data"])

print()