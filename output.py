from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from typing import Union, Literal
from pydantic import BaseModel, Field
from config import QWEN_API_KEY as apikey, QWEN_BASE_URL as burl
from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model="qwen3.8-max",
    temperature=0.7,
    api_key=apikey,
    base_url=burl,
    max_tokens=10000,
    timeout=30,
    max_retries=3,
    extra_body={"enable_thinking": False},   # 关闭 Qwen 思考模式，允许 tool_choice
)

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    gender: str = Field(description="人物性别")
    occupation: str = Field(description="人物职业")
    location: str = Field(description="人物所在地")

class WeatherInfo(BaseModel):
    """天气信息"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="当前温度")
    condition: str = Field(description="天气状况,如晴天/多云/下雨")

class OtherInfo(BaseModel):
    """兜底类型：输入不属于任何已知类型时输出这个"""
    category: Literal["other"] = "other"   # 固定值，明确标记不属于已知类型
    reason: str = Field(description="为什么无法匹配已知类型")


agent = create_agent(
    model=model,
    response_format=ToolStrategy(
        schema=Union[Person, WeatherInfo, OtherInfo],   # 三选一，含兜底
        tool_message_content="提取完成",
        handle_errors=True,
    ),
)

# 两个不同类型的输入，验证模型会分别选择不同的结构
for query in [
    "我想了解一个人物信息,他的姓名是张三,年龄是20岁,性别是男,职业是程序员,所在地是北京",
    "上海现在的温度是28度,多云天气",
    "帮我推荐一部适合新手看的编程书",   # 无关输入：和 Person/WeatherInfo 都没关系
]:
    print(f"\n=== 输入: {query[:20]}... ===")
    result = agent.invoke({"messages": [query]})
    sr = result.get("structured_response")
    print(f"输出类型: {type(sr).__name__}")
    print(f"输出内容: {sr}")
