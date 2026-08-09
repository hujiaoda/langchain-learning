import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from config import QWEN_BASE_URL as Qurl, QWEN_API_KEY as Qapi

# 读取本地图片，转 base64
with open("b_3b4986251de35e8d2bcf5cfa5066bdbc.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")
image_url = f"data:image/jpeg;base64,{image_base64}"

model = ChatOpenAI(
    model="qwen3.7-plus",
    api_key=Qapi,
    base_url=Qurl,
)

msg = HumanMessage(content=[
    {"type": "text", "text": "这个图片里面有啥"},
    {"type": "image_url", "image_url": {"url": image_url}},
])

response = model.invoke([msg])
print(response.content)