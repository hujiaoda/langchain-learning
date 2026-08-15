"""
中间件演示：完整代码 + 运行流程

本文件演示 langchain 1.x 的中间件机制：
- @before_model / @after_model 装饰器创建中间件
- 中间件被框架"自动调用"（不是我们自己调的）
- 用打印日志证明调用时机
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl


# ── 一个工具（让 agent 有事可做）──
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"{city}: 晴天 25°C"


# ── 两个中间件：只做一件事——打印"我被调用了" ──
@before_model
def log_before_model(state, runtime):
    print("  [中间件] 模型调用前 → 我被框架自动调用了")
    return None   # 返回 None = 不干预，放行


@after_model
def log_after_model(state, runtime):
    print("  [中间件] 模型调用后 → 我又被框架自动调用了")
    return None


# ── 构建 agent（挂上中间件）──
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=apikey,
    base_url=burl,
    timeout=30,
)
agent = create_agent(
    model,
    tools=[get_weather],
    middleware=[log_before_model, log_after_model],
)

# ── 运行 ──
print("=== agent 开始执行 ===")
result = agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
print("=== agent 执行结束 ===")
print()
print("最终回答:", result["messages"][-1].content)