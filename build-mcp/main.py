import weather
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('weather query', port=8001)
mcp.add_tool(weather.query_weather)


def main():
    asyncio.run(mcp.run_streamable_http_async())


if __name__ == "__main__":
    main()
