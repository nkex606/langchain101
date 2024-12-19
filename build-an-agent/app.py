import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<langchain_api_key>"
os.environ["TAVILY_API_KEY"] = "<tavily_api_key>"
os.environ["OPENAI_API_KEY"] = "<openai_api_key>"

# Create the agent
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o")
search = TavilySearchResults(max_results=2)

# use Tavily search engine to search
# search_results = search.invoke("What's the latest chip for macbook pro in 2024?")
# print(search_results)

tools = [search]

# model with tool calling
# 當這個問題預期要使用到 tool, 會發現結果沒有任何 content, 但會有個 ToolCalls 結果
# bind_tools 並不會使用到 agent, 只是告訴 model 可以使用 tool

# model_with_tools = model.bind_tools(tools)
# resp = model_with_tools.invoke([HumanMessage(content="What's the latest chip for macbook pro in 2024?")])
# print(f"ContentString: {resp.content}")
# print(f"ToolCalls: {resp.tool_calls}")

# create agent
# 傳入 model 而不是 model_with_tools, create_react_agent func 會呼叫 bind_tools
# agent is stateless, 如果要讓 agent 可以知道上一個問題，需加入 checkpointer
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# thread_id 讓 agent 知道要從哪個 thread/conversation 回復
config = {"configurable": {"thread_id": "abc123"}}

# Use the agent
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im John! and i live in Taichung, Taiwan")]},
    config,
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weathder where I live?")]}, config
):
    print(chunk)
    print("----")
