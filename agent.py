import sys


# Windows 终端默认 GBK 编码，遇到 emoji 会报错，统一改成 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl,TAVILY_API_KEY as Tapi
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from rich import print as rprint
from tavily import TavilyClient
@tool
def get_weather(city:str)->str:
    """获取城市天气"""
    return f"城市{city}的天气是晴天"

search = DuckDuckGoSearchRun()
@tool
def search_web(query:str)->str:
    """搜索网络"""
    return search.run(query)

Tcilent=TavilyClient(
    api_key=Tapi,
)
@tool
def web_search(qury:str)->str:
    """搜索网络,返回整合后的搜索结果"""
    res = Tcilent.search(qury,max_results=2)
    return str(res)

agent = create_agent(
    model=init_chat_model(
        model="deepseek-v4-flash",
        model_provider="deepseek",
        api_key=apikey,
        base_url=burl,
    ),
    tools=[get_weather,web_search],
    debug=True,
)

response = agent.invoke({
    "messages":[
        #HumanMessage(content="简单说下什么是transformer模型")
        #HumanMessage("帮我看下北京天气如何")
        HumanMessage(
            content="https://openai.com/zh-Hans-CN/form/codex-for-oss/,帮我看看这个网站说的什么,还有什么样的项目才算满足要求？不要回答太复杂"
        )
    ]
})
rprint(response)