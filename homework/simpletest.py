from dotenv import load_dotenv
import asyncio
load_dotenv()

from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.llms.openai import OpenAI
tools = []

ll = "asdasd"

def add(a:float,b:float)->float:
    print(ll)
    return a + b

def multiply(a:float,b:float)->float:
    return a * b

tools.append(FunctionTool.from_defaults(
    add,
    name="add",
    description="sum of two float number"
))

tools.append(FunctionTool.from_defaults(
    multiply,
    name="multiply",
    description="Product of two float number"
))


llm = OpenAI(model="o3-mini", temperature=0.7)

workflow = FunctionAgent(
    llm=llm,
    system_prompt="You are a helpful AI assistant that can perform precise arithmetic by calling the function add(a: int, b: int) and multiply(a：float, b:float). Whenever the user asks you to compute the sum of two numbers, invoke the add function with those two integers as arguments. After the function returns a result, present the numeric answer directly to the user. If the user asks anything else, respond normally without calling the function.",
    tools=tools
)

async def chat(llm, query):
    response = await llm.run(query)
    return response

result = asyncio.run(chat(workflow, "18加27乘2等于多少"))
print(result)