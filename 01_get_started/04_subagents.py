import os
from typing import Literal

from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from deepagents import create_deep_agent

BASE_URL = os.getenv("OPENAI_BASE_URL")      # 例如 https://api.your-service.com/v1
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

llm1 = ChatOpenAI(
    model=MODEL_NAME,           # 模型名，会透传给服务端
    base_url=BASE_URL,          # 自定义 base_url
    api_key=API_KEY,            # 你的 API key
    temperature=0.5
    # max_tokens=..., timeout=..., 其他参数也可以直接写
)

llm2 = ChatOpenAI(
    model=MODEL_NAME,           # 模型名，会透传给服务端
    base_url=BASE_URL,          # 自定义 base_url
    api_key=API_KEY,            # 你的 API key
    temperature=0.7
    # max_tokens=..., timeout=..., 其他参数也可以直接写
)

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": llm2,  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    model=llm1,
    subagents=subagents
)

result = agent.invoke({"messages": [{"role": "user", "content": "如何深度思考？"}]})

# Print the agent's response
print(result["messages"][-1].content)
