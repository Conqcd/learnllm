import pandas as pd
from pathlib import Path

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import asyncio
import duckdb
load_dotenv()
import DataAgent
from DataAgent import agent


async def fuck():
    # Example usage
    response = await agent.workflow.run("请帮我分析一下这个数据")
    print(response)

print(asyncio.run(fuck()))


conn = duckdb.connect('mydata.db')
print("yes")