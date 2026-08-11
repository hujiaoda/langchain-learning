from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl


# ── 工具定义 ──
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


# 工具名字 → 函数本体，替代 if/else
tools_map = {t.name: t for t in [get_weather, calculator, get_time]}


# ── 记忆系统（不变）──
def get_role(m):
    if isinstance(m, dict):
        return m["role"]
    return m.type

def get_content(m):
    if isinstance(m, dict):
        return m["content"]
    return m.content

def trim_messages(messages, n):
    system = [m for m in messages if get_role(m) == "system"]
    others = [m for m in messages if get_role(m) != "system"]
    return system + others[-n:]


# ── 模型 + 工具绑定 ──
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
model_with_tools = model.bind_tools([get_weather, calculator, get_time])


# ── 初始化对话 ──
chat_prompt_template = ChatPromptTemplate([
    ("system", "你是一个{name}，可以查天气、算数学、报时间"),
    ("human", "你好,你叫什么名字"),
    ("ai", "我是一个{name}没有名字"),
    ("human", "这样啊,我是{user_input}"),
])
full_history = chat_prompt_template.invoke({
    "name": "猫娘助手",
    "user_input": "鸡蛋(eku)",
}).messages  # 取消息列表


# ── 主循环 ──
while True:
    user_input = input("You: ")
    if user_input == "quit":
        break

    # 追加用户消息
    full_history.append({"role": "user", "content": user_input})

    # Agent 循环：模型可能多次调工具
    while True:
        memory = trim_messages(full_history, n=12)   # 工具调用内部消息多，放宽一点
        response = model_with_tools.invoke(memory)

        # 有 tool_call → 执行工具，继续循环
        if response.tool_calls:
            full_history.append(response)           # 保存这次工具调用决策
            for tc in response.tool_calls:
                tool_fn = tools_map[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                full_history.append({               # 保存工具执行结果
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            # 继续循环，模型看到结果后决定是否继续调工具
        else:
            # 最终自然语言回复
            print(response.content)
            full_history.append({"role": "assistant", "content": response.content})
            break