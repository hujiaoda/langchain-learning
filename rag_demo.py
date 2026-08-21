"""
RAG 三阶段学习 demo
==================================================================
阶段1: 手动 RAG —— 手写整条链路（加载→向量化→检索→拼接→生成），理解每步原理
阶段2: 完整管线 —— 真实文件加载 + 文本切分 + 带评分检索，理解"为什么切分"
阶段3: Agent 化 RAG —— 检索封装成工具让 AI 自主调用，结合已学的 create_agent 与流式输出

环境依赖（全部已装，无需额外安装）：
- embedding 用 Qwen text-embedding-v3（langchain_openai 走 dashscope 兼容地址）
- 生成用 deepseek-v4-flash
- 向量库用 langchain_core 内置 InMemoryVectorStore（官方教程同款）

运行：python rag_demo.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from config import DEEPSEEK_API_KEY, BASE_URL, QWEN_API_KEY, QWEN_BASE_URL


class DashScopeEmbeddings(Embeddings):
    """极简 embedding 封装：直接调 dashscope 的 OpenAI 兼容接口。

    为什么不用 langchain_openai.OpenAIEmbeddings？
    它默认开启 tiktoken 分词（为了按 token 数做长度安全切分），
    发送的 input 是 token id 数组 list[list[int]]。
    OpenAI 官方接口接受这种格式，但 dashscope 兼容接口只认
    str / list[str]，收到数字数组会报 400（实测踩坑）。
    手写 30 行反而透明可控——这就是 Embeddings 接口的全部契约：
    embed_documents(批量文本) + embed_query(单个查询)。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 学习用途：打印调用轨迹，让你看到 add_documents 内部何时调用了这里
        print(f"    [调用] DashScopeEmbeddings.embed_documents 被触发，"
              f"向量化 {len(texts)} 条文本，首条: {texts[0][:18]}...")
        resp = self.client.embeddings.create(model=self.model, input=texts)
        print(f"    [调用] 服务器返回，每条得到一个 1024 维向量")
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        # 学习用途：打印调用轨迹
        print(f"    [调用] DashScopeEmbeddings.embed_query 被触发，向量化查询: {text[:18]}...")
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding


# ---------------------------------------------------------------- 公共组件
def build_embeddings():
    """embedding 模型：文本 -> 向量。

    走 Qwen 的 OpenAI 兼容接口（dashscope），不需要代理、国内直连。
    text-embedding-v3 把每段文本变成 1024 维向量，
    语义相近的文本向量在空间中距离近。
    """
    return DashScopeEmbeddings(
        model="text-embedding-v3",
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )


def build_llm():
    """生成模型：检索到资料后负责回答问题。"""
    return init_chat_model(
        model="deepseek-v4-flash",
        model_provider="deepseek",
        api_key=DEEPSEEK_API_KEY,
        base_url=BASE_URL,
        timeout=30,
    )


# ================================================================ 阶段一：手动 RAG
def stage1_manual_rag():
    print("=" * 70)
    print("阶段1: 手动 RAG —— 手写整条链路（最简）")
    print("=" * 70)

    # 1. 加载：手工构造 3 条知识（真实场景这里是读文件/网页）
    knowledge = [
        "LangChain 是一个用于构建大语言模型应用的框架，"
        "提供统一的模型调用接口、提示词模板、工具绑定等抽象，"
        "把模型调用、数据处理、Agent 编排封装成可组合的模块。",
        "Agent 是 LangChain 的核心概念。Agent 不是简单的模型调用，"
        "而是让模型自主决定调用哪些工具、按什么顺序执行。"
        "典型流程：模型根据用户问题生成工具调用指令，"
        "执行工具后把结果返回给模型继续推理，直到得出最终答案。",
        "记忆系统分短期记忆和长期记忆。"
        "短期记忆用 checkpointer 保存对话历史，通过 thread_id 区分会话；"
        "长期记忆用 Store 保存跨会话的持久信息，"
        "工具通过 InjectedStore 注入访问长期记忆。",
    ]
    docs = [Document(page_content=t) for t in knowledge]
    print(f"[1/4] 已加载 {len(docs)} 条知识")

    # 2. 向量化 + 存储：每段文本 -> 向量，存进内存向量库
    embeddings = build_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(docs)
    print(f"[2/4] 已向量化并存库，库中共 {len(vector_store.store)} 条")

    # 3. 检索：把"问题"也向量化，找语义最近的 k 段文本
    question = "agent 是怎么工作的？"
    hits = vector_store.similarity_search(question, k=2)
    print(f"[3/4] 检索 '{question}'，命中 {len(hits)} 段：")
    for i, d in enumerate(hits):
        print(f"      hit{i}: {d.page_content[:40]}...")

    # 4. 生成：把检索结果拼进 prompt，让模型基于资料回答
    #    这就是 RAG 的 G：模型不凭记忆瞎编，而是"基于检索到的证据"回答
    context = "\n\n".join(d.page_content for d in hits)
    messages = [
        SystemMessage(
            "你是知识库问答助手。只依据用户提供的资料回答问题，"
            "资料中没有的信息直接说不知道，不要编造。"
        ),
        HumanMessage(f"资料：\n{context}\n\n问题：{question}\n\n请回答："),
    ]
    answer = build_llm().invoke(messages)
    print(f"[4/4] 模型回答：\n{answer.content}\n")

    # 5. 对照实验：不检索直接问（无 RAG 场景）
    answer_no_rag = build_llm().invoke(
        [HumanMessage("agent 是怎么工作的？（不要使用任何资料）")]
    )
    print("对照：不检索直接问模型的回答（可能基于训练数据、未必准确）：")
    print(f"{answer_no_rag.content}\n")


# ================================================================ 阶段二：完整管线
# 示例知识库文件：模拟"项目文档"，写多主题长文，用于演示切分
DOC_FILE = "kb_docs.txt"

KB_CONTENT = """LangChain 框架入门
LangChain 是一个用于构建大语言模型应用的框架，提供统一的模型调用接口、提示词模板、工具绑定等抽象。
它把模型调用、数据处理、Agent 编排封装成可组合的模块，让开发者用几行代码就能搭建复杂的 LLM 应用。
LangChain 的核心设计哲学是"组件可组合"，模型、提示词、工具、记忆都是独立的组件，可以自由搭配。

Agent 智能体
Agent 是 LangChain 的核心概念，它不是简单的模型调用，而是让模型自主决定调用哪些工具、按什么顺序执行。
典型流程是：模型根据用户问题生成工具调用指令，执行工具后把结果返回给模型继续推理。
Agent 的决策能力来自模型本身，工具只是给模型提供的"手和脚"。
当问题涉及多个步骤时，Agent 会把任务拆解成多轮工具调用来完成。

记忆系统
记忆系统分为短期记忆和长期记忆。短期记忆用 checkpointer 保存对话历史，通过 thread_id 区分会话。
长期记忆用 Store 保存跨会话的持久信息，比如用户的名字、偏好等。
工具可以通过 InjectedStore 参数注入访问长期记忆，AI 在对话中自主决定何时读写记忆。
短期记忆是"会话内"的记忆，长期记忆是"跨会话"的记忆，这是两者的本质区别。

结构化输出
LangChain 用 Pydantic 模型定义输出结构，让模型返回符合 schema 的 JSON。
结构化输出通常通过工具策略实现：模型被要求调用一个"输出工具"，工具的参数就是目标结构。
如果模型返回的结构不符合定义，框架会自动重试纠正。
结构化输出的价值是把模型的自由文本变成程序可用的数据，直接进入下游业务逻辑。
"""


def stage2_full_pipeline():
    print("=" * 70)
    print("阶段2: 完整管线 —— 文件加载 + 切分 + 带评分检索")
    print("=" * 70)

    # 1. 加载：把内置文本写入文件，再像真实场景一样读文件
    #    真实项目中这里是 TextLoader（一行包装 open() + Document）
    with open(DOC_FILE, "w", encoding="utf-8") as f:
        f.write(KB_CONTENT)
    with open(DOC_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    doc = Document(page_content=raw, metadata={"source": DOC_FILE})
    print(f"[1/5] 加载文档 {len(raw)} 字符（作为 1 个 Document）")

    # 2. 切分：一整个文档没法检索（太长，命中粒度太粗），切成小块
    #    chunk_size=200：每块约 200 字符；chunk_overlap=40：相邻块重叠 40 字符，
    #    重叠是为了避免"一句话被拦腰截断在边界"导致语义丢失
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents([doc])
    print(f"[2/5] 切分成 {len(chunks)} 个 chunk（chunk_size=200, overlap=40）")
    for i, c in enumerate(chunks[:5]):
        print(f"      chunk[{i}] ({len(c.page_content)}字): {c.page_content[:30]}...")

    # 3. 向量化存储
    vector_store = InMemoryVectorStore(build_embeddings())
    vector_store.add_documents(chunks)
    print(f"[3/5] 已入库 {len(vector_store.store)} 个 chunk")

    # 4. 带评分检索：similarity_search_with_score 返回余弦相似度
    #    注意：score 越大越相关（1.0 = 完全相似），这是刚翻源码确认的
    for question in ["checkpointer 的作用是什么？", "为什么结构化输出要用 Pydantic？"]:
        print(f"[4/5] 检索：'{question}'")
        for d, score in vector_store.similarity_search_with_score(question, k=2):
            print(f"      score={score:.4f} | {d.page_content[:45]}...")

    # 5. 生成：命中多块拼起来回答
    question = "短期记忆和长期记忆有什么区别？"
    hits = vector_store.similarity_search(question, k=2)
    context = "\n\n".join(f"[来源 {i}] {d.page_content}" for i, d in enumerate(hits))
    messages = [
        SystemMessage("你是知识库问答助手，只依据资料回答，资料没有的就说不知道。"),
        HumanMessage(f"资料：\n{context}\n\n问题：{question}"),
    ]
    answer = build_llm().invoke(messages)
    print(f"[5/5] 回答：\n{answer.content}\n")


# ================================================================ 阶段三：Agent 化 RAG
def stage3_agent_rag():
    print("=" * 70)
    print("阶段3: Agent 化 RAG —— 检索封装成工具，AI 自主调用 + 流式输出")
    print("=" * 70)

    # 1. 复用阶段二的文档与向量库（这里直接重新建，保持脚本可独立运行）
    doc = Document(page_content=KB_CONTENT, metadata={"source": DOC_FILE})
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=40
    ).split_documents([doc])
    vector_store = InMemoryVectorStore(build_embeddings())
    vector_store.add_documents(chunks)
    print(f"[1/3] 向量库就绪，共 {len(vector_store.store)} 个 chunk")

    # 2. 把"检索"封装成工具：工具 = 给 AI 的能力扩展。
    #    区别在于：阶段一是"我们替 AI 检索"，这里是"AI 自己决定要不要检索、检索什么"。
    #    这样 AI 可以自主判断"这问题我知识库里有吗？查一下"。
    @tool
    def search_knowledge(query: str) -> str:
        """从知识库检索与 query 最相关的段落（返回前 3 段原文，带来源编号）。

        Args:
            query: 用户问题的检索关键词或完整问题。
        """
        hits = vector_store.similarity_search(query, k=3)
        return "\n\n".join(
            f"[来源{i}] {d.page_content}" for i, d in enumerate(hits)
        )

    agent = create_agent(
        build_llm(),
        tools=[search_knowledge],
        system_prompt=(
            "你是知识库问答助手。回答前先调用 search_knowledge 检索相关资料，"
            "严格基于检索到的内容回答，资料里没有的就说不知道。"
        ),
    )

    # 3. 流式输出：stream_mode="messages" 逐 token 流式（stream.py 同款写法）
    print("[2/3] 开始流式对话（AI 会先自主检索再回答）")
    question = "LangChain 的核心设计哲学是什么？"
    print(f"[3/3] 问题：{question}")
    print("回答：", end="", flush=True)
    for msg_chunk, _meta in agent.stream(
        {"messages": [HumanMessage(question)]}, stream_mode="messages"
    ):
        # msg_chunk 是 AIMessageChunk：content 是文本增量，tool_call_chunks 是工具调用增量
        if getattr(msg_chunk, "content", ""):
            print(msg_chunk.content, end="", flush=True)
        for tc in getattr(msg_chunk, "tool_call_chunks", []) or []:
            if tc.get("name"):
                print(f"\n      [AI 调用工具] {tc['name']}(query=...)", end="", flush=True)
    print("\n")


# ---------------------------------------------------------------- 入口
if __name__ == "__main__":
    stage1_manual_rag()
    stage2_full_pipeline()
    stage3_agent_rag()
    # 清理示例文件
    if os.path.exists(DOC_FILE):
        os.remove(DOC_FILE)
    print("三个阶段全部完成。")