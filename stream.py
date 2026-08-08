#from pyexpat import model
from langchain.chat_models import init_chat_model
import os
from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl

def trim_messages(messages, n):
    """保留 system + 最后 n 条消息"""
    # 找出所有 system 消息
    system = [m for m in messages if m["role"] == "system"]

    # 找出所有非 system 消息，只保留最后 n 条
    others = [m for m in messages if m["role"] != "system"]
    others = others[-n:]  # 切片：从倒数第 n 条开始

    return system + others


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
# 完整历史：永远不删
full_history = [{"role": "system", "content": "你是一个猫娘,回答简短点"}]

while True:
    user_input = input("You: ")
    if user_input == "quit":
        break

    # 完整历史追加用户消息
    full_history.append({"role": "user", "content": user_input})

    # 裁剪后喂给模型
    memory = trim_messages(full_history, n=6)

    full_response = ""
    for chunk in model.stream(memory):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content
    print()

    # 完整历史追加 AI 回复
    full_history.append({"role": "assistant", "content": full_response})