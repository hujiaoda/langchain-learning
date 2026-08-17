"""
工具访问长期记忆：完整可运行示例

核心机制：
- create_agent(..., store=store) 把长期记忆挂给 agent
- 工具声明 store: BaseStore 参数 → 框架自动注入同一个 store
- AI 在对话中自动决定何时读写记忆
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from typing_extensions import Annotated
from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl

# ① 创建长期记忆存储
store = InMemoryStore()


# ② 两个记忆工具：InjectedStore 标注 = "langgraph 在运行时注入 store，不给 LLM 看"
@tool
def write_memory(key: str, value: str, *, store: Annotated[BaseStore, InjectedStore]) -> str:
    """记住用户的重要信息（名字、偏好、事实），key 用语义名"""
    store.put(("memories",), key, {"value": value})
    return f"已记住 {key}"


@tool
def read_memory(key: str, *, store: Annotated[BaseStore, InjectedStore]) -> str:
    """读取用户之前告诉你的信息"""
    item = store.get(("memories",), key)
    return item.value["value"] if item else f"没有关于 {key} 的记忆"


# ③ 挂上 store + 工具
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=apikey,
    base_url=burl,
    timeout=30,
)
agent = create_agent(
    model,
    tools=[write_memory, read_memory],
    store=store,
    system_prompt=(
        "你是一个有长期记忆的助手。"
        "用户可能之前告诉过你信息（名字、偏好等）。"
        "回答涉及用户个人信息时，先调用 read_memory 查询；"
        "如果用户告诉你新的重要信息，调用 write_memory 记住。"
    ),
)

# ④ 验证：对话中 AI 自动写记忆
print("=== 对话1: 告诉它名字，观察 AI 是否自动调用 write_memory ===")
r1 = agent.invoke({"messages": [HumanMessage("我叫胡椒，请记住我")]})
print("AI:", r1["messages"][-1].content)

# 查看 store 里实际存了什么
print()
print("=== store 里的记忆（search 查看）===")
for item in store.search(("memories",)):
    print(f"  key={item.key}, value={item.value}")

# ⑤ 验证：再次问，AI 从 store 读取
print()
print("=== 对话2: 问名字（没有 checkpointer，新会话，但 store 还在）===")
r2 = agent.invoke({"messages": [HumanMessage("我叫什么名字？")]})
print("AI:", r2["messages"][-1].content)