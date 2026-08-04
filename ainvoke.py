from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY as apikey,BASE_URL as burl
import time
import asyncio
#model=openai.ChatOpenAI(api_key=apikey,base_url=burl)
model=ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=apikey,
    base_url=burl,
)

async def demo_async_invoke():
    print("==== 演示：ainvoke异步（非阻塞）效果 ===")
    start_time =time.perf_counter()#记录开始时间
    print("开始...")
    #1.创建任务Task
    print(">>>发起异步模型调用")
    async_task=asyncio.create_task(model.ainvoke("一句话解释AI"))

    #2.执行其他任务
    print(">>请求已发送,继续执行逻辑....")
    for i in range(3):
        await asyncio.sleep(1)
        print(f">>>正在执行第{i+1}个任务...已耗时{time.perf_counter()-start_time}秒")
    #3.获取模型结果
    print(">>>本地任务完成,检查模型状态...")
    response=await async_task

    end_time=time.perf_counter()
    print(f">>>模型返回:{response.content}")
    print(f"===总运行耗时：{end_time-start_time:.2f}s====")
# async def async_stream(message):
#     async for chunk in model.astream(message):
#         print(chunk.content, end="", flush=True)
#     print()

# async def main():
#     prompts = [
#         "帮我解释下什么是人工智能,简短点",
#         "帮我解释下什么是机器学习,简短点",
#         "帮我解释下什么是深度学习,简短点",
#     ]

#     # 异步：3 个请求同时发出，总耗时 ≈ 最慢那个
#     start_time = time.time()
#     await asyncio.gather(*(async_stream(p) for p in prompts))
#     print(f"异步总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    #asyncio.run(main())
    asyncio.run(demo_async_invoke())