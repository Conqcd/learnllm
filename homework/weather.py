from fastmcp import FastMCP
import asyncio

# 监听 12111 端口，服务名随意
mcp = FastMCP('weather-service', port=12111)

@mcp.tool()
def query_weather(city: str) -> str:
    """Return the weather of city."""
    print(city)
    return "晴天"

print("Mounted routes:")
app = mcp.streamable_http_app()
for route in app.routes:
    print(" •", getattr(route, "path", route))

if __name__ == '__main__':
    # 使用 SSE 模式启动
    asyncio.run(mcp.run_stdio_async())