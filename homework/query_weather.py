import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
load_dotenv()
# 1. 指定 MCP 服务地址
mcp_url = "http://127.0.0.1:12111/mcp/"
client = BasicMCPClient(mcp_url)

# 2. 获取远程工具列表
tool_spec = McpToolSpec(client=client)

async def fetch_tools(tool_spec):
    tools = await tool_spec.to_tool_list_async()
    return tools

tools = asyncio.get_event_loop().run_until_complete(fetch_tools(tool_spec))

# 3. 配置 LLM 并注入 functions schema
chat_llm = OpenAI(
    model_name="o3-mini",
    temperature=0,
)


messages = [
    ChatMessage(role="system",    content="你是一个天气助手。可以调用工具：query_weather 来查询天气。"),
    ChatMessage(role="user",      content="上海今天天气是?"),
]

workflow = FunctionAgent(llm=chat_llm,tools=tools, system_prompt="你是一个天气助手。可以调用工具：query_weather 来查询天气。")

# 5. 调用远端 查询天气

async def chat(llm:FunctionAgent, query:str):
    response = await llm.run(query)
    return response

async def chat2(llm:OpenAI, query):
    response = llm.chat_with_tools(messages=query,tools=tools)
    return response

response = asyncio.run(chat(workflow, "上海今天天气是?"))
print("Agent Response:", response)
response = asyncio.run(chat2(chat_llm, messages))

print("Agent Response:", response)
