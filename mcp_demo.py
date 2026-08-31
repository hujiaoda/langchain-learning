"""一键跑 MCP 演示：先手动 client，再 agent 自动调工具

用法（VS Code 集成终端，别按 F5）：
    python mcp_demo.py
"""
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent
PYTHON = sys.executable


def run(title: str, filename: str):
    print(f"\n{'=' * 56}\n>>> {title}\n{'=' * 56}", flush=True)
    subprocess.run([PYTHON, str(BASE / filename)], cwd=BASE)


if __name__ == "__main__":
    run("1. client 手动发现并调用 server 工具", "mcp_demo_client.py")
    run("2. agent 自动发现并调用 server 工具", "mcp_agent_demo.py")
