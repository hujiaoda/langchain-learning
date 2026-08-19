"""embedding 到底是什么：把一次真实的 embedding API 调用完整拆开。

重点：程序里的一切代码，本质都是"发 HTTP 请求 + 收响应"。
这个脚本打印出发出去什么、收回来什么，让你亲眼看到数据结构。
"""
import sys
import json
import math

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from config import QWEN_API_KEY, QWEN_BASE_URL

# ============================================================
# 第 0 步：client 是什么
# client = OpenAI SDK 里的一个"客户端对象"。它只有一件事：
# 帮你把"调用"翻译成 HTTP 请求发到服务器。
# 它本身不计算任何东西，只是个"发信员"。
# ============================================================
client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

# ============================================================
# 第 1 步：发一次请求
# resp = client.embeddings.create(model=..., input=[...])
# 这一行实际发生的事：
#   1. 你的电脑拼一个 HTTP 请求（JSON 文本）：
#      POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
#      {"model": "text-embedding-v3", "input": ["猫", ...]}
#   2. 通过网络发给阿里云服务器
#   3. 服务器用 text-embedding-v3 模型算出向量
#   4. 服务器把结果拼成 JSON 响应发回来
#   5. OpenAI SDK 把响应包装成 resp 对象返回给你
# ============================================================
resp = client.embeddings.create(
    model="text-embedding-v3",
    input=["猫", "我喜欢猫", "上证指数大涨"],
)

print("=== resp 到底是什么 ===")
print("类型:", type(resp).__name__)  # Embedding 对象
print()

# 把 resp 对象转成字典，原样打印（这就是服务器发回来的 JSON）
data = resp.model_dump()
print("=== 服务器返回的完整 JSON（截断显示）===")
text = json.dumps(data, ensure_ascii=False)
print(text[:700])
print("......(中间省略)......")
print()

print("=== resp 的结构解读 ===")
print(f"resp.data: 一个列表，长度 = {len(resp.data)}（等于你输入了几条文本）")
for i, d in enumerate(resp.data):
    print(f"  data[{i}] 里有:")
    print(f"    index     = {d.index}   # 第几条输入")
    print(f"    embedding = 长度 {len(d.embedding)} 的一串浮点数，前6个: {d.embedding[:6]}")
print()

print("=== embedding 有什么实际意义？===")
v0, v1, v2 = [d.embedding for d in resp.data]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


print(f"cosine相似度  '猫'        vs '我喜欢猫'    = {cos(v0, v1):.4f}  ← 语义相近，相似度高")
print(f"cosine相似度  '猫'        vs '上证指数大涨' = {cos(v0, v2):.4f}  ← 语义无关，相似度低")
print()
print("结论: embedding 就是『把一段文本变成一个位置坐标』，")
print("语义越像的文本，坐标越近。检索就是在坐标空间里找最近的邻居。")
print()

print("=== 回到 rag_demo.py 那两行 ===")
print("embed_documents: [d.embedding for d in resp.data]")
print("  = 把 resp.data 里每一条的 embedding 抽出来，组成列表返回")
print("embed_query: resp.data[0].embedding")
print("  = 只输入了一条，取第 0 条（唯一那条）的 embedding")