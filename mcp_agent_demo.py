"""MCP + LangChain：把 MCP server 的工具接进 create_agent，让模型自己决定调用"""
import asyncio
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl


async def main():
    server_path = Path(__file__).parent / "mcp_demo_server.py"
    params = StdioServerParameters(command=sys.executable, args=[str(server_path)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 关键一步：MCP 工具 → LangChain 工具（适配器做"翻译"）
            tools = await load_mcp_tools(session)
            print("接进来的工具:", [t.name for t in tools])

            model = init_chat_model(
                model="deepseek-v4-flash",
                model_provider="deepseek",
                temperature=0.7,
                api_key=apikey,
                base_url=burl,
            )
            agent = create_agent(model, tools=tools)

            resp = await agent.ainvoke(
                {"messages": [HumanMessage("用工具算一下 1+2 等于几？")]}
            )
            print("回答:", resp["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
