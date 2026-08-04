#from pyexpat import model
from langchain.chat_models import init_chat_model
import os
from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl


model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    temperature=0.7,
    api_key=apikey,
    base_url=burl,
    max_tokens=1000,
    timeout=30,
    max_retries=3,
)
message=[{"role": "system", "content": "你是一个猫娘,回答简短点"}]
while True:
    user_input=input("You: ")
    if user_input=="quit":
        break
    message.append({"role": "user", "content": user_input})
    response=model.invoke(message)
    print(response.content)
    message.append({"role": "assistant", "content": response.content})
    