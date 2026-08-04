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
    max_tokens=10000,
    timeout=30,
    max_retries=3,
)
message=[{"role": "system", "content": "你是一个猫娘,回答简短点"}]
#for i in range(1):
while True:
    user_input=input("You: ")
    if user_input=="quit":
        break
    message.append({"role": "user", "content": user_input})
    full_response=""
    for chunk in model.stream(message):
        print(chunk.content,end="",flush=True)
        full_response+=chunk.content
    # response=model.invoke(message)
    # print(response.content)
    message.append({"role": "assistant", "content": full_response})
    
    """ 
   #获取响应元数据
    metadata=response.response_metadata
    print(f"使用的模型:{metadata['model_name']}")
    print(f"结束原因:{metadata['finish_reason']}")
    print(f"模型供应商:{metadata['model_provider']}")
    #获取Token使用情况
    usage=metadata.get('token_usage',{})
    print(f"输入tokens:{usage.get('prompt_tokens')}")
    #print(f"输出tokens:{usage.get('prompt_tokens')}")
    print(f"输出tokens:{usage.get('completion_tokens')}")
    print(f"总结tokens:{usage.get('total_tokens')}")
    #获取信息ID
    print(f"消息ID：{response.id}")
    """