"""MCP 最小示例：一个跑在 stdio 上的 server，提供三个工具

注意：环境里 mcp 是 1.x（langchain-mcp-adapters 要求 mcp<2），
所以用 v1 的 FastMCP 写法；mcp 2.x 里它改名为 MCPServer。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime

from mcp.server.fastmcp import FastMCP

from tavily import TavilyClient
from config import TAVILY_API_KEY as Tapi
client = TavilyClient(api_key=Tapi)
server = FastMCP("demo-server")


@server.tool()
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b


@server.tool()
def current_time() -> str:
    """返回当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@server.tool()
def web_search(query: str) -> str:
    """搜索网页，返回前 2 条结果"""
    response = client.search(query=query, max_results=2)
    return f"搜索结果：{str(response)}"

if __name__ == "__main__":
    # stdio 传输：client 以子进程方式启动本文件，通过 stdin/stdout 通信
    server.run(transport="stdio")
