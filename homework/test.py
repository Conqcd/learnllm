import pandas as pd
from pathlib import Path

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import asyncio
import duckdb
load_dotenv()
from os import getenv
import DataAgent
from DataAgent import agent


# async def fuck():
#     # Example usage
#     response = await agent.workflow.run("请帮我分析一下这个数据")
#     print(response)
#
# print(asyncio.run(fuck()))
#
#
# conn = duckdb.connect('mydata.db')
# print("yes")

from agno.agent import Agent
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai.like import OpenAILike
from agno.storage.sqlite import SqliteStorage
from rich.pretty import pprint

# UserId for the memories
user_id = "ava"
# Database file for memory and storage
db_file = "tmp/agent.db"

# Initialize memory.v2
memory = Memory(
    # Use any model for creating memories
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
)
# Initialize storage
storage = SqliteStorage(table_name="agent_sessions", db_file=db_file)

# Initialize Agent
memory_agent = Agent(
    model=OpenAILike(id="o3-mini",
                     api_key=getenv("OPENAI_API_KEY"),
                     base_url=getenv("OpenAI_API_BASE"),
                     ),
    # Store memories in a database
    memory=memory,
    # Give the Agent the ability to update memories
    enable_agentic_memory=True,
    # OR - Run the MemoryManager after each response
    enable_user_memories=True,
    # Store the chat history in the database
    storage=storage,
    # Add the chat history to the messages
    add_history_to_messages=True,
    # Number of history runs
    num_history_runs=3,
    markdown=True,
)

memory.clear()
memory_agent.print_response(
    "My name is Ava and I like to ski.",
    user_id=user_id,
    stream=True,
    stream_intermediate_steps=True,
)
print("Memories about Ava:")
pprint(memory.get_user_memories(user_id=user_id))

memory_agent.print_response(
    "I live in san francisco, where should i move within a 4 hour drive?",
    user_id=user_id,
    stream=True,
    stream_intermediate_steps=True,
)
print("Memories about Ava:")
pprint(memory.get_user_memories(user_id=user_id))