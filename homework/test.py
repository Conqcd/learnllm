import pandas as pd
from pathlib import Path

import workflows
from agno.utils.log import log_debug, logger
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
agent.workflow.tools[0].metadata.to_openai_tool()
print(asyncio.run(fuck()))
