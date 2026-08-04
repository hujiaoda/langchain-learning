from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl


#model=openai.ChatOpenAI(api_key=apikey,base_url=burl)
model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=apikey,
    base_url=burl,
)
messages=[
    "帮我解释下什么是人工智能,简短点",
    "今天晚上吃什么好呢,推荐一下,简短点",
    "你怎么看待现在大环境,简短点"
]
response=model.batch(messages)
for response in response:
    print(response.content)
