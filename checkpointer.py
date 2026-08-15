import sys
# Windows 终端默认 GBK 编码，遇到 emoji 会报错，统一改成 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl
from rich import print as rprint
checkpointer = InMemorySaver()

agent = create_agent(
    model=init_chat_model(
        model="deepseek-v4-flash",
        model_provider="deepseek",
        temperature=0.7,
        api_key=apikey,
        base_url=burl,
        max_tokens=10000,
        timeout=30,
        max_retries=3,
    ),
    tools=[],
    checkpointer=checkpointer,
)
config={"configurable": {"thread_id": "session-1"}}
# 关键：两次 invoke 用同一个 thread_id，checkpointer 才能恢复历史
response1 = agent.invoke(
    {"messages": [HumanMessage("我是胡椒")]},
    config=config,
)
print(f"Agent: {response1['messages'][-1].content}")

response2 = agent.invoke(
    {"messages": [HumanMessage("我是谁")]},
    config=config,
)
print(f"Agent: {response2['messages'][-1].content}")


print("=================================================")

store = InMemoryStore()

# namespace 必须是"完整的一个元组"（注意逗号！("users") 是字符串，("users",) 才是元组）
# user_id 是 namespace 的一部分，不是独立参数
store.put(("users", "user-1"), "name", {"name": "小A"})
item = store.get(("users", "user-1"), "name")
rprint(item)

# delete：删除某条记忆（记错了/过期/用户要求删除）
store.delete(("users", "user-1"), "name")
rprint("删除后:", store.get(("users", "user-1"), "name"))   # → None

# search：用"前缀"列出该分区下所有条目
store.put(("users", "user-1"), "pref", {"value": "简洁"})
store.put(("users", "user-2"), "name", {"name": "小B"})
rprint("users 下所有条目:", store.search(("users",)))
