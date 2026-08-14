"""
智能助手 SmartAssistant
"""
import sys
import ast
import operator
# Windows 终端默认 GBK 编码，遇到 emoji 会报错，统一改成 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from tavily import TavilyClient
from config import (
    DEEPSEEK_API_KEY as apikey,
    BASE_URL as burl,
    TAVILY_API_KEY as Tapi,
    QWEN_API_KEY as QwenApi,
    QWEN_BASE_URL as qwen_burl,
)


# ── 工具：Tavily 网络搜索 ──
tavily_client = TavilyClient(api_key=Tapi)

@tool
def web_search(query: str) -> str:
    """搜索网络，返回整合后的搜索结果"""
    result = tavily_client.search(query, max_results=3)
    return str(result)

# 允许的运算节点 → 对应操作（白名单）
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def safe_calc(expression: str) -> str:
    """安全计算数学表达式：只允许数字和四则运算。

    和 eval 的区别：解析成 AST 后逐个节点校验，只接受白名单内的节点类型。
    函数调用、属性访问、导入、下标等一切可能执行代码的节点直接拒绝。
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"表达式语法错误: {expression}")

    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](eval_node(node.operand))
        # 任何其他节点类型都拒绝
        raise ValueError(f"不支持的表达式元素: {type(node).__name__}")

    try:
        return str(eval_node(tree.body))
    except (ZeroDivisionError, OverflowError) as e:
        raise ValueError(f"计算错误: {e}")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '123*456' 或 '(2+3)*4'"""
    return safe_calc(expression)

class SmartAssistant:
    def __init__(self):
        # 模型
        self.model = init_chat_model(
            model="deepseek-v4-flash",
            model_provider="deepseek",
            temperature=0.7,
            api_key=apikey,
            base_url=burl,
            max_tokens=10000,
            timeout=30,
            max_retries=3,
        )
        # agent（带 Tavily 搜索工具）
        self.agent = create_agent(self.model, tools=[web_search, calculator])
        # 记忆：摘要 + 最近消息（混合式）
        self.summary = ""          # 早期对话的压缩摘要
        self.recent = []           # 最近的原始消息（保留细节）
        self.MAX_RECENT = 8        # recent 条数阈值，超过就压缩

    def _trim_summarize(self):
        """混合记忆：recent 超阈值时，把最老一半总结进 summary，只留最近一半"""
        overflow = self.recent[: len(self.recent) // 2]
        self.recent = self.recent[len(self.recent) // 2 :]

        # 溢出消息转成文本
        lines = []
        for m in overflow:
            who = "用户" if isinstance(m, HumanMessage) else "AI"
            lines.append(f"{who}: {m.content}")
        overflow_text = "\n".join(lines)

        # 调模型：旧摘要 + 新增对话 → 新摘要
        prompt = (
            "你是对话摘要器。请把【旧摘要】和【新增对话】合并成一份新的简洁摘要，"
            "保留关键信息（用户意图、重要事实、约定），不要超过 200 字。\n\n"
            f"【旧摘要】\n{self.summary or '（无）'}\n\n"
            f"【新增对话】\n{overflow_text}"
        )
        self.summary = self.model.invoke(prompt).content

    def chat(self, message: str):
        """对话：流式逐 token 输出（生成器，调用方用 for 消费）"""
        self.recent.append(HumanMessage(content=message))
        if len(self.recent) > self.MAX_RECENT:
            self._trim_summarize()

        # 组装给 agent 的消息：system(摘要) + 最近原文
        system_prompt = (
            f"对话摘要：{self.summary or '（暂无）'}\n"
            "你是一个智能助手，必要时可以用网络搜索获取最新信息"
        )
        messages = [SystemMessage(content=system_prompt)] + self.recent

        full_reply = ""
        for msg_chunk, _ in self.agent.stream(
            {"messages": messages}, stream_mode="messages"
        ):
            # 过滤工具结果 chunk：messages 模式会把 ToolMessage 也流出来，只保留模型输出
            if isinstance(msg_chunk, ToolMessage):
                continue
            if msg_chunk.content:
                full_reply += msg_chunk.content
                yield msg_chunk.content

        # 流式结束后，把完整回答记入历史，供下一轮使用
        self.recent.append(AIMessage(content=full_reply))
        if len(self.recent) > self.MAX_RECENT:
            self._trim_summarize()

    def chat_structured(self, schema, message: str):
        """结构化输出：从消息中提取 schema 结构，返回 Pydantic 对象。

        用 Qwen 模型：deepseek 思考模式不支持 tool_choice=required，
        Qwen 关闭思考模式后 ToolStrategy 可用。
        response_format 在创建 agent 时固定，因此每个 schema 单独构建 agent。
        """
        qwen_model = ChatOpenAI(
            model="qwen3.8-max",
            temperature=0,
            api_key=QwenApi,
            base_url=qwen_burl,
            max_tokens=10000,
            timeout=30,
            max_retries=3,
            extra_body={"enable_thinking": False},
        )
        structured_agent = create_agent(
            qwen_model,
            tools=[],
            response_format=ToolStrategy(schema=schema, handle_errors=True),
        )
        result = structured_agent.invoke({"messages": [HumanMessage(content=message)]})
        return result.get("structured_response")


if __name__ == "__main__":
    assistant = SmartAssistant()
    while True:
        user_input = input("You: ")
        if user_input == "quit":
            break
        print("AI: ", end="", flush=True)
        for token in assistant.chat(user_input):
            print(token, end="", flush=True)
        print()