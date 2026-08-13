import sys
# Windows 终端默认 GBK 编码，遇到 emoji 会报错，统一改成 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl

# ── 工具定义（不变）──
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气，返回温度和天气状况"""
    weather_db = {"北京": "晴天, 25°C", "上海": "多云, 28°C", "深圳": "暴雨, 22°C"}
    return weather_db.get(city, f"{city}: 晴天, 20°C")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，返回计算结果"""
    return str(eval(expression))

@tool
def get_time(place: str) -> str:
    """查询指定城市当前的日期和时间"""
    from datetime import datetime
    return f"{place}当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# ── 模型 + agent（替代原来的手写循环）──
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    temperature=0.7,
    api_key=apikey,
    base_url=burl,
    max_tokens=10000,
    timeout=30,
    max_retries=3,
)

agent = create_agent(model, tools=[get_weather, calculator, get_time])

# ── 多轮记忆：手动维护消息历史 ──
# create_agent 每次 invoke 是独立的，不跨轮记忆，所以自己维护消息列表
history = [SystemMessage(content="你是一个猫娘助手，可以查天气、算数学、报时间")]

# ── 记忆截断：只保留最后 n 条非 system 消息，防止上下文无限增长 ──
# 原版是 get_role() 判断裸字典，新版直接用 isinstance 判断消息类型
def trim_history(messages, n=10):
    system = [m for m in messages if isinstance(m, SystemMessage)]
    others = [m for m in messages if not isinstance(m, SystemMessage)]
    return system + others[-n:]

# ── 主循环 ──
while True:
    user_input = input("You: ")
    if user_input == "quit":
        break

    history.append(HumanMessage(content=user_input))

    # 流式输出：stream_mode="messages" 让 agent 像 model 一样逐 token 输出
    print("AI: ", end="", flush=True)
    full_reply = ""
    for msg_chunk, _ in agent.stream({"messages": trim_history(history)}, stream_mode="messages"):
        if msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
            full_reply += msg_chunk.content
    print()

    # 把最终回答记入历史，供下一轮使用
    history.append(AIMessage(content=full_reply))
