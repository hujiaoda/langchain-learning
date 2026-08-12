from pydantic import BaseModel, Field
from config import QWEN_API_KEY as apikey, QWEN_BASE_URL as burl
from langchain_openai import ChatOpenAI
from rich import print as rprint


model = ChatOpenAI(
    model="qwen3.8-max",
    temperature=0.7,
    api_key=apikey,
    base_url=burl,
    max_tokens=10000,
    timeout=30,
    max_retries=3,
)

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    gender: str = Field(description="人物性别")
    occupation: str = Field(description="人物职业")
    location: str = Field(description="人物所在地")

model_with_structured_output = model.with_structured_output(Person)
result = model_with_structured_output.invoke("我想了解一个人物信息,他的姓名是张三,年龄是20岁,性别是男,职业是程序员,所在地是北京")
rprint(result)
