#from pyexpat import model
from langchain.chat_models import init_chat_model
import os
from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl
from langchain_core.prompts import ChatPromptTemplate

from rich import print as rprint

def get_role(m):
    """兼容字典和消息对象，返回 role 字符串"""
    if isinstance(m, dict):
        return m["role"]
    return m.type  # SystemMessage / HumanMessage / AIMessage

def get_content(m):
    """兼容字典和消息对象，返回 content"""
    if isinstance(m, dict):
        return m["content"]
    return m.content

def trim_messages(messages, n):
    """保留 system + 最后 n 条消息"""
    system = [m for m in messages if get_role(m) == "system"]
    others = [m for m in messages if get_role(m) != "system"]
    return system + others[-n:]


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

#full_history = [{"role": "system", "content": "你是一个猫娘,回答简短点"}]

chat_prompt_template=ChatPromptTemplate(
    [
        ("system","你是一个{name}可以回答任何问题"),
        ("human","你好,你叫什么名字"),
        ("ai","我是一个{name}没有名字"),
        ("human","这样啊,我是{user_input}")
    ]
)
result = chat_prompt_template.invoke({"name":"猫娘","user_input":"鸡蛋(eku)"})
# 完整历史：取出 messages 列表
full_history = result.messages

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