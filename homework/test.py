import json
import tempfile
import csv
import pandas as pd
from pathlib import Path

import workflows
from agno.utils.log import log_debug, logger
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import duckdb

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

class DataFunctionAgent():
    def __init__(self,
        db_path: Optional[str] = None,
        read_only: bool = False,
        init_commands: Optional[List] = None,
        config: Optional[dict] = None,
        **kwargs):
        llm = OpenAI(model="qwen3-235b-a22b", temperature=0.7)
        self.workflow = FunctionAgent(
            llm=llm,
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
            **kwargs
        )
        self._connection = None
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.base_dir = Path.cwd()
        self.db_path: Optional[str] = db_path
        self.read_only: bool = read_only
        self.config: Optional[dict] = config
        self.init_commands: Optional[List] = init_commands

    def create_pandas_dataframe(
        self, dataframe_name: str, create_using_function: str, function_parameters: Dict[str, Any]
    ) -> str:
        """Creates a pandas dataframe named `dataframe_name` by running a function `create_using_function` with the parameters `function_parameters`.
        Returns the created dataframe name as a string if successful, otherwise returns an error message.

        For Example:
        - To create a dataframe `csv_data` by reading a CSV file, use: {"dataframe_name": "csv_data", "create_using_function": "read_csv", "function_parameters": {"filepath_or_buffer": "data.csv"}}
        - To create a dataframe `csv_data` by reading a JSON file, use: {"dataframe_name": "json_data", "create_using_function": "read_json", "function_parameters": {"path_or_buf": "data.json"}}

        :param dataframe_name: The name of the dataframe to create.
        :param create_using_function: The function to use to create the dataframe.
        :param function_parameters: The parameters to pass to the function.
        :return: The name of the created dataframe if successful, otherwise an error message.
        """
        try:
            log_debug(f"Creating dataframe: {dataframe_name}")
            log_debug(f"Using function: {create_using_function}")
            log_debug(f"With parameters: {function_parameters}")

            if dataframe_name in self.dataframes:
                return f"Dataframe already exists: {dataframe_name}"

            # Create the dataframe
            dataframe = getattr(pd, create_using_function)(**function_parameters)
            if dataframe is None:
                return f"Error creating dataframe: {dataframe_name}"
            if not isinstance(dataframe, pd.DataFrame):
                return f"Error creating dataframe: {dataframe_name}"
            if dataframe.empty:
                return f"Dataframe is empty: {dataframe_name}"
            self.dataframes[dataframe_name] = dataframe
            log_debug(f"Created dataframe: {dataframe_name}")
            return dataframe_name
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            return f"Error creating dataframe: {e}"

    def run_dataframe_operation(self, dataframe_name: str, operation: str, operation_parameters: Dict[str, Any]) -> str:
        """Runs an operation `operation` on a dataframe `dataframe_name` with the parameters `operation_parameters`.
        Returns the result of the operation as a string if successful, otherwise returns an error message.

        For Example:
        - To get the first 5 rows of a dataframe `csv_data`, use: {"dataframe_name": "csv_data", "operation": "head", "operation_parameters": {"n": 5}}
        - To get the last 5 rows of a dataframe `csv_data`, use: {"dataframe_name": "csv_data", "operation": "tail", "operation_parameters": {"n": 5}}

        :param dataframe_name: The name of the dataframe to run the operation on.
        :param operation: The operation to run on the dataframe.
        :param operation_parameters: The parameters to pass to the operation.
        :return: The result of the operation if successful, otherwise an error message.
        """
        try:
            log_debug(f"Running operation: {operation}")
            log_debug(f"On dataframe: {dataframe_name}")
            log_debug(f"With parameters: {operation_parameters}")

            # Get the dataframe
            dataframe = self.dataframes.get(dataframe_name)

            # Run the operation
            result = getattr(dataframe, operation)(**operation_parameters)

            log_debug(f"Ran operation: {operation}")
            try:
                try:
                    return result.to_string()
                except AttributeError:
                    return str(result)
            except Exception:
                return "Operation ran successfully"
        except Exception as e:
            logger.error(f"Error running operation: {e}")
            return f"Error running operation: {e}"

    def save_file(self, contents: str, file_name: str, overwrite: bool = True) -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        :param contents: The contents to save.
        :param file_name: The name of the file to save to.
        :param overwrite: Overwrite the file if it already exists.
        :return: The file name if successful, otherwise returns an error message.
        """
        try:
            file_path = self.base_dir.joinpath(file_name)
            logger.debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(contents)
            logger.info(f"Saved: {file_path}")
            return str(file_name)
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """
        Returns the duckdb connection

        :return duckdb.DuckDBPyConnection: duckdb connection
        """
        if self._connection is None:
            connection_kwargs: Dict[str, Any] = {}
            if self.db_path is not None:
                connection_kwargs["database"] = self.db_path
            if self.read_only:
                connection_kwargs["read_only"] = self.read_only
            if self.config is not None:
                connection_kwargs["config"] = self.config
            self._connection = duckdb.connect(**connection_kwargs)
            try:
                if self.init_commands is not None:
                    for command in self.init_commands:
                        self._connection.sql(command)
            except Exception as e:
                logger.exception(e)
                logger.warning("Failed to run duckdb init commands")

        return self._connection
    def show_tables(self, show_tables: bool) -> str:
        """Function to show tables in the database

        :param show_tables: Show tables in the database
        :return: List of tables in the database
        """
        if show_tables:
            stmt = "SHOW TABLES;"
            tables = self.run_query(stmt)
            logger.debug(f"Tables: {tables}")
            return tables
        return "No tables to show"

    def describe_table(self, table: str) -> str:
        """Function to describe a table

        :param table: Table to describe
        :return: Description of the table
        """
        stmt = f"DESCRIBE {table};"
        table_description = self.run_query(stmt)

        logger.debug(f"Table description: {table_description}")
        return f"{table}\n{table_description}"

    def inspect_query(self, query: str) -> str:
        """Function to inspect a query and return the query plan. Always inspect your query before running them.

        :param query: Query to inspect
        :return: Query plan
        """
        stmt = f"explain {query};"
        explain_plan = self.run_query(stmt)

        logger.debug(f"Explain plan: {explain_plan}")
        return explain_plan

    def run_query(self, query: str) -> str:
        """Function that runs a query and returns the result.

        :param query: SQL query to run
        :return: Result of the query
        """

        # -*- Format the SQL Query
        # Remove backticks
        formatted_sql = query.replace("`", "")
        # If there are multiple statements, only run the first one
        formatted_sql = formatted_sql.split(";")[0]

        try:
            logger.info(f"Running: {formatted_sql}")

            query_result = self.connection.sql(formatted_sql)
            result_output = "No output"
            if query_result is not None:
                try:
                    results_as_python_objects = query_result.fetchall()
                    result_rows = []
                    for row in results_as_python_objects:
                        if len(row) == 1:
                            result_rows.append(str(row[0]))
                        else:
                            result_rows.append(",".join(str(x) for x in row))

                    result_data = "\n".join(result_rows)
                    result_output = ",".join(query_result.columns) + "\n" + result_data
                except AttributeError:
                    result_output = str(query_result)

            logger.debug(f"Query result: {result_output}")
            return result_output
        except duckdb.ProgrammingError as e:
            return str(e)
        except duckdb.Error as e:
            return str(e)
        except Exception as e:
            return str(e)

    def summarize_table(self, table: str) -> str:
        """Function to compute a number of aggregates over a table.
        The function launches a query that computes a number of aggregates over all columns,
        including min, max, avg, std and approx_unique.

        :param table: Table to summarize
        :return: Summary of the table
        """
        table_summary = self.run_query(f"SUMMARIZE {table};")

        logger.debug(f"Table description: {table_summary}")
        return table_summary

    def get_table_name_from_path(self, path: str) -> str:
        """Get the table name from a path

        :param path: Path to get the table name from
        :return: Table name
        """
        import os

        # Get the file name from the path
        file_name = path.split("/")[-1]
        # Get the file name without extension from the path
        table, extension = os.path.splitext(file_name)
        # If the table isn't a valid SQL identifier, we'll need to use something else
        table = table.replace("-", "_").replace(".", "_").replace(" ", "_").replace("/", "_")

        return table

    def create_table_from_path(self, path: str, table: Optional[str] = None, replace: bool = False) -> str:
        """Creates a table from a path

        :param path: Path to load
        :param table: Optional table name to use
        :param replace: Whether to replace the table if it already exists
        :return: Table name created
        """

        if table is None:
            table = self.get_table_name_from_path(path)

        logger.debug(f"Creating table {table} from {path}")
        create_statement = "CREATE TABLE IF NOT EXISTS"
        if replace:
            create_statement = "CREATE OR REPLACE TABLE"

        create_statement += f" '{table}' AS SELECT * FROM '{path}';"
        self.run_query(create_statement)
        logger.debug(f"Created table {table} from {path}")
        return table

    def export_table_to_path(self, table: str, format: Optional[str] = "PARQUET", path: Optional[str] = None) -> str:
        """Save a table in a desired format (default: parquet)
        If the path is provided, the table will be saved under that path.
            Eg: If path is /tmp, the table will be saved as /tmp/table.parquet
        Otherwise it will be saved in the current directory

        :param table: Table to export
        :param format: Format to export in (default: parquet)
        :param path: Path to export to
        :return: None
        """
        if format is None:
            format = "PARQUET"

        logger.debug(f"Exporting Table {table} as {format.upper()} to path {path}")
        if path is None:
            path = f"{table}.{format}"
        else:
            path = f"{path}/{table}.{format}"
        export_statement = f"COPY (SELECT * FROM {table}) TO '{path}' (FORMAT {format.upper()});"
        result = self.run_query(export_statement)
        logger.debug(f"Exported {table} to {path}/{table}")
        return result

agent = DataFunctionAgent()

def create_pandas_dataframe(dataframe_name: str, create_using_function: str, function_parameters: Dict[str, Any]
) -> str:
    """Creates a pandas dataframe named `dataframe_name` by running a function `create_using_function` with the parameters `function_parameters`.
    Returns the created dataframe name as a string if successful, otherwise returns an error message.

    For Example:
    - To create a dataframe `csv_data` by reading a CSV file, use: {"dataframe_name": "csv_data", "create_using_function": "read_csv", "function_parameters": {"filepath_or_buffer": "data.csv"}}
    - To create a dataframe `csv_data` by reading a JSON file, use: {"dataframe_name": "json_data", "create_using_function": "read_json", "function_parameters": {"path_or_buf": "data.json"}}

    :param dataframe_name: The name of the dataframe to create.
    :param create_using_function: The function to use to create the dataframe.
    :param function_parameters: The parameters to pass to the function.
    :return: The name of the created dataframe if successful, otherwise an error message.
    """
    return agent.workflow.create_pandas_dataframe(dataframe_name, create_using_function, function_parameters)

def run_dataframe_operation(dataframe_name: str, operation: str, operation_parameters: Dict[str, Any]) -> str:
    """Runs an operation `operation` on a dataframe `dataframe_name` with the parameters `operation_parameters`.
    Returns the result of the operation as a string if successful, otherwise returns an error message.

    For Example:
    - To get the first 5 rows of a dataframe `csv_data`, use: {"dataframe_name": "csv_data", "operation": "head", "operation_parameters": {"n": 5}}
    - To get the last 5 rows of a dataframe `csv_data`, use: {"dataframe_name": "csv_data", "operation": "tail", "operation_parameters": {"n": 5}}

    :param dataframe_name: The name of the dataframe to run the operation on.
    :param operation: The operation to run on the dataframe.
    :param operation_parameters: The parameters to pass to the operation.
    :return: The result of the operation if successful, otherwise an error message.
    """
    return agent.workflow.create_pandas_dataframe(dataframe_name, operation, operation_parameters)


def save_file(contents: str, file_name: str, overwrite: bool = True) -> str:
    """Saves the contents to a file called `file_name` and returns the file name if successful.

    :param contents: The contents to save.
    :param file_name: The name of the file to save to.
    :param overwrite: Overwrite the file if it already exists.
    :return: The file name if successful, otherwise returns an error message.
    """
    return agent.workflow.save_file(contents, file_name, overwrite)


def show_tables(show_tables: bool) -> str:
    """Function to show tables in the database

    :param show_tables: Show tables in the database
    :return: List of tables in the database
    """
    return agent.workflow.show_tables(show_tables)

def describe_table(table: str) -> str:
    """Function to describe a table

    :param table: Table to describe
    :return: Description of the table
    """
    return agent.workflow.describe_table(table)

def inspect_query(query: str) -> str:
    """Function to inspect a query and return the query plan. Always inspect your query before running them.

    :param query: Query to inspect
    :return: Query plan
    """
    return agent.workflow.inspect_query(query)

def run_query(query: str) -> str:
    """Function that runs a query and returns the result.

    :param query: SQL query to run
    :return: Result of the query
    """
    return agent.workflow.run_query(query)

def create_table_from_path(self, path: str, table: Optional[str] = None, replace: bool = False) -> str:
    """Creates a table from a path

    :param path: Path to load
    :param table: Optional table name to use
    :param replace: Whether to replace the table if it already exists
    :return: Table name created
    """
    return agent.workflow.create_table_from_path(path, table, replace)

def summarize_table(self, table: str) -> str:
    """Function to compute a number of aggregates over a table.
    The function launches a query that computes a number of aggregates over all columns,
    including min, max, avg, std and approx_unique.

    :param table: Table to summarize
    :return: Summary of the table
    """
    return agent.workflow.summarize_table(table)

def export_table_to_path(self, table: str, format: Optional[str] = "PARQUET", path: Optional[str] = None) -> str:
    """Save a table in a desired format (default: parquet)
    If the path is provided, the table will be saved under that path.
        Eg: If path is /tmp, the table will be saved as /tmp/table.parquet
    Otherwise it will be saved in the current directory

    :param table: Table to export
    :param format: Format to export in (default: parquet)
    :param path: Path to export to
    :return: None
    """
    return agent.workflow.export_table_to_path(table, format, path)


tools: List[Any] = []
tools.append(create_pandas_dataframe)
tools.append(run_dataframe_operation)
tools.append(save_file)
tools.append(show_tables)
tools.append(describe_table)
tools.append(inspect_query)
tools.append(run_query)
tools.append(create_table_from_path)
tools.append(summarize_table)
tools.append(export_table_to_path)

agent.workflow.tools = tools