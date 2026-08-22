"""交叉编码（Cross-Encoder / Rerank）真实演示。

用阿里百炼 qwen3-rerank（真正的交叉编码模型）对比双塔余弦检索。
验证歧义场景下交叉编码的精度优势。

前置：.env 里需要 WORKSPACE_ID=你的百炼业务空间ID（控制台 → 业务空间管理）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math

import requests
from openai import OpenAI
from config import QWEN_API_KEY, DASHSCOPE_API_KEY

# ============================================================
# 交叉编码：问题 + 每条文档拼一起过模型，直接打分
# 官方接口：https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks
# ============================================================
WORKSPACE_ID = os.getenv("WORKSPACE_ID")


def rerank(query, documents):
    """调用 qwen3-rerank：返回 [(index, score), ...] 按分数降序"""
    if not WORKSPACE_ID:
        print("错误：.env 里缺少 WORKSPACE_ID（百炼业务空间ID）")
        raise SystemExit(1)
    url = (
        f"https://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com"
        "/compatible-api/v1/reranks"
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen3-rerank",
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        },
        timeout=30,
    )
    resp.raise_for_status()
    # qwen3-rerank 响应：results 在顶层
    results = resp.json()["results"]
    return sorted(results, key=lambda r: r["relevance_score"], reverse=True)


# ============================================================
# 双塔（embedding 余弦）：对照组
# ============================================================
client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL = "text-embedding-v3"


def bi_encoder_search(query, documents):
    """双塔：问题/文档各自向量化，算余弦，按相似度降序"""
    texts = [query] + documents
    vecs = [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts).data]
    qv, doc_vecs = vecs[0], vecs[1:]

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))

    scored = sorted(((cos(qv, v), i) for i, v in enumerate(doc_vecs)), reverse=True)
    return scored


if __name__ == "__main__":
    query = "苹果公司的股价最近怎么样？"
    documents = [
        "苹果公司发布了新一代 iPhone，销量创历史新高。",      # 相关
        "苹果是常见的水果，富含维生素 C。",                    # 歧义陷阱（水果）
        "特斯拉股价近期波动较大，马斯克减持了股份。",         # 无关（但词面含"股价"）
        "苹果公司市值突破三万亿，股价创历史新高。",          # 高度相关
    ]

    print("问题:", query)
    print()

    print("【双塔 embedding 余弦检索】")
    for score, i in bi_encoder_search(query, documents):
        print(f"  {score:.4f}  {documents[i]}")
    print()

    print("【交叉编码 qwen3-rerank 精排】")
    for r in rerank(query, documents):
        print(f"  {r['relevance_score']:.4f}  {documents[r['index']]}")
    print()

    print("看点：双塔被'特斯拉(含股价)'词面误导排到第3；交叉编码应把它压到最低")