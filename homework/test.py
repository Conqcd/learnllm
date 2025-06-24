import streamlit as st
import json
import tempfile
import csv
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from phi.agent.duckdb import DuckDbAgent
from agno.tools.pandas import PandasTools
semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                }
            ]
        }
duckdb_agent = DuckDbAgent(
            semantic_model=json.dumps(semantic_model),
            markdown=True,
            add_history_to_messages=False,  # Disable chat history
            followups=False,  # Disable follow-up queries
            read_tool_call_history=False,  # Disable reading tool call history
            )
tools = duckdb_agent.get_tools()