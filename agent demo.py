"""
Agent 进阶 —— 从手动循环到 langgraph

核心问题：你已经手写了 Agent 循环（stream.py），为什么还需要 langgraph？
答案：手动循环像手写链表——能跑，但状态一复杂就失控。
     langgraph 本质是一个"显式状态机"，你把状态和边定义好，它帮你跑。
"""

from config import DEEPSEEK_API_KEY as apikey, BASE_URL as burl
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# 用 print 替代 rprint，避免 LLM 回复中的 emoji 在 Windows 终端报编码错
def p(obj):
    s = str(obj)
    # Windows GBK 终端扛不住 emoji，过滤掉
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("gbk", errors="replace").decode("gbk"))

# ============================================================
# 准备：共享的工具和模型（三个阶段都用）
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    db = {"北京": "晴天, 25°C", "上海": "多云, 28°C", "深圳": "暴雨, 22°C"}
    return db.get(city, f"{city}: 晴天, 20°C")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '123*456'"""
    return str(eval(expression))

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，返回相关知识"""
    db = {
        "python": "Python 由 Guido van Rossum 于 1991 年发布，是一门解释型、动态类型语言。",
        "langchain": "LangChain 是一个 LLM 应用开发框架，提供 chains、agents、RAG 等抽象。",
        "gpu": "GPU（图形处理器）最初用于图形渲染，现广泛用于深度学习训练和推理。",
    }
    for key, val in db.items():
        if key in query.lower():
            return val
    return f"未找到关于 '{query}' 的知识。已知: {list(db.keys())}"

TOOLS = [get_weather, calculator, search_knowledge]
TOOLS_MAP = {t.name: t for t in TOOLS}

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
model_with_tools = model.bind_tools(TOOLS)


# ============================================================
# 阶段1：手动 Agent 循环（复习，和 stream.py 一回事）
# ============================================================
# 流程：用户输入 → LLM 决定是否调工具 → 调了就把结果喂回去 → 循环直到 LLM 直接回复

def manual_agent(user_input: str, max_turns: int = 5):
    """手动实现的 Agent 循环 —— 这就是 langgraph 底层在做的事"""
    messages = [SystemMessage(content="你是一个助手，可以查天气、算数学、搜索知识。")]
    messages.append(HumanMessage(content=user_input))

    for turn in range(max_turns):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            # LLM 想调工具 → 执行工具，结果追加到消息
            for tc in response.tool_calls:
                tool_fn = TOOLS_MAP[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            # 继续循环，让 LLM 看到工具结果后再决定
        else:
            # LLM 直接回复 → 结束
            return response.content

    return "达到最大轮次，Agent 未能在限制内完成任务。"

p("[bold]阶段1: 手动 Agent 循环[/bold]")
result = manual_agent("北京天气怎么样？顺便帮我算 123*456")
p(f"  结果: {result}")


# ============================================================
# 阶段2：create_react_agent —— 一行代码替你写循环
# ============================================================
# langgraph 把"LLM调用 → 检查tool_calls → 执行工具 → 喂回去 → 循环"
# 这个模式封装成了 create_react_agent。
# 它不是黑箱，内部就是个状态机（阶段3会拆开看）。

from langchain.agents import create_agent

agent = create_agent(model, TOOLS)

p("\n[bold]阶段2: create_react_agent (langgraph 封装)[/bold]")
result = agent.invoke({
    "messages": [HumanMessage(content="深圳天气怎么样？GPU是什么？")]
})
# agent.invoke 返回的是完整状态，最后一条 AI 消息就是最终回复
final_msg = result["messages"][-1]
p(f"  结果: {final_msg.content}")


# ============================================================
# 阶段3：手写 StateGraph —— 拆开 langgraph 的黑箱
# ============================================================
# create_react_agent 本质上就是个状态图：
#
#   [开始] → LLM节点 → 有tool_calls? → 工具节点 → 回到LLM节点
#                        ↓ 没有
#                      [结束]
#
# 每个节点是一个函数，边是条件判断。下面手写一遍。

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 状态定义：Agent 在运行中需要记住的所有东西
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages 是 langgraph 的消息合并函数

# 节点1：调用 LLM
def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 节点2：执行工具
def call_tools(state: AgentState):
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        tool_fn = TOOLS_MAP[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": results}

# 路由函数：LLM 回复后，判断是去工具节点还是结束
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"       # 有工具调用 → 去工具节点
    return END              # 没有 → 结束

# 构建图
graph = StateGraph(AgentState)
graph.add_node("llm", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("llm")                              # 入口：先调 LLM
graph.add_conditional_edges("llm", should_continue, {     # LLM 之后：条件路由
    "tools": "tools",
    END: END,
})
graph.add_edge("tools", "llm")                            # 工具之后：回到 LLM
app = graph.compile()

p("\n[bold]阶段3: 手写 StateGraph[/bold]")
p("  图结构: llm → (有tool) → tools → llm → (无tool) → END")
result = app.invoke({
    "messages": [HumanMessage(content="上海天气？langchain是什么？")]
})
final_msg = result["messages"][-1]
p(f"  结果: {final_msg.content}")
p(f"  消息数: {len(result['messages'])} 条（包含 LLM 调用和工具结果）")