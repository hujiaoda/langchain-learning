import os
from re import A
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl
from rich import print as rprint
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model

model=ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=apikey,
    base_url=burl,
)

model2 = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    temperature=0.7,
    api_key=apikey,
    base_url=burl,
    max_tokens=10000,
    configurable_fields=("model","model_provider","temperature","max_tokens"),
)
config={
    "run_name":"joke_generation",#Langsmith运行名字
    "tags":["my_tag1","mytag2"],
    "metadate":{
        "user_id":"shkstart",#记录用户id
        "session_id":"sess_123"#记录会话ID
    },
    "configurable":{
        "model":"deepseek-v4-pro",#配置模型参数
        "model_provider":"openai",#配置模型提供商参数
        "temperature":0.7,#配置温度参数
        "max_token":10000#最大token数
    }
}


response=model2.invoke("你是谁",config=config)
rprint(response)

#rprint(model.profile)
