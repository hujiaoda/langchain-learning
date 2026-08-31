"""MCP 最小示例：client 连接 server，发现工具并调用"""
import asyncio
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # 让 client 用"当前解释器"启动 server 子进程，走 stdin/stdout 通信
    # 注意：路径要以"client 脚本所在目录"为基准，不能用相对路径（取决于启动目录）
    server_path = Path(__file__).parent / "mcp_demo_server.py"
    params = StdioServerParameters(command=sys.executable, args=[str(server_path)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 第 1 步：发现工具（client 问 server：你有什么工具？）
            tools = await session.list_tools()
            print("=== server 提供的工具 ===")
            for t in tools.tools:
                print(f"- {t.name}: {t.description}")

            # 第 2 步：调用工具（client 替模型执行 server 里的函数）
            print("\n=== 调用 add(1, 2) ===")
            r1 = await session.call_tool("add", {"a": 1, "b": 2})
            print("结果:", r1.content[0].text)

            print("\n=== 调用 current_time() ===")
            r2 = await session.call_tool("current_time", {})
            print("结果:", r2.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
