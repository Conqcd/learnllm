import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
import subprocess

# 1. 指定 MCP 服务地址
mcp_url = "http://127.0.0.1:12111/mcp/"
client = BasicMCPClient(mcp_url)

client2 = BasicMCPClient(
    "python",                      # 可执行命令
    args=["weather.py"],  # 要传给它的脚本名
)
# 2. 获取远程工具列表
tool_spec = McpToolSpec(client=client2)

async def fetch_tools(tool_spec):
    tools = await tool_spec.fetch_tools()
    return tools

tools = asyncio.get_event_loop().run_until_complete(fetch_tools(tool_spec))

# 3. 配置 LLM 并注入 functions schema
chat_llm = OpenAI(
    model_name="gpt-3.5-turbo-0613",
    temperature=0,
    functions=[t.to_openai_schema() for t in tools],
    function_call="auto",
)


# 5. 调用远端 查询天气

async def chat(llm, query):
    response = await llm.run(query)
    return response

response = asyncio.run(chat(chat_llm,"上海今天天气是?"))

print("Agent Response:", response)
