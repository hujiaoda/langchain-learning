"""
结构化输出进阶 —— 从"格式化"到"提取"

核心问题：LLM 返回的是字符串，你后续代码怎么处理？
答案：Pydantic 把 LLM 输出变成类型化对象，后续代码直接 .字段名 访问。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from config import QWEN_API_KEY as apikey, QWEN_BASE_URL as burl
from langchain_openai import ChatOpenAI
from rich import print as rprint

model = ChatOpenAI(
    model="qwen3.8-max",
    temperature=0,
    api_key=apikey,
    base_url=burl,
    max_tokens=10000,
    timeout=30,
    max_retries=3,
)

# ============================================================
# 阶段1：嵌套结构 —— 真实数据很少是扁平的
# ============================================================

class Skill(BaseModel):
    """技能"""
    name: str = Field(description="技能名称")
    level: Literal["初级", "中级", "高级", "专家"] = Field(description="熟练程度")

class WorkExperience(BaseModel):
    """工作经历"""
    company: str = Field(description="公司名称")
    position: str = Field(description="职位")
    years: float = Field(description="工作年数")
    highlights: List[str] = Field(description="主要成就, 至少1条", min_length=1)

class Resume(BaseModel):
    """简历 —— 嵌套了 Skill 和 WorkExperience"""
    name: str = Field(description="姓名")
    email: Optional[str] = Field(description="邮箱, 可能没有", default=None)
    skills: List[Skill] = Field(description="技能列表")
    experiences: List[WorkExperience] = Field(description="工作经历列表")
    summary: str = Field(description="一句话总结")


# 一段"脏"的非结构化文本，模拟从 PDF/网页扒下来的
raw_text = """
张三，邮箱 zhangsan@example.com。
精通 Python（高级）和 React（中级），还会一点 Docker（初级）。
在字节跳动做了2.5年前端开发，主导了组件库重构，将构建时间缩短了60%。
之前在美团做了1年后端开发，负责订单系统微服务拆分。
"""

resume_model = model.with_structured_output(Resume)
resume = resume_model.invoke(f"从以下文本中提取简历信息:\n{raw_text}")
rprint("[bold]阶段1: 嵌套结构[/bold]")
rprint(f"  姓名: {resume.name}")
rprint(f"  邮箱: {resume.email}")
rprint(f"  技能数: {len(resume.skills)}")
for s in resume.skills:
    rprint(f"    - {s.name} ({s.level})")
rprint(f"  经历数: {len(resume.experiences)}")
for e in resume.experiences:
    rprint(f"    - {e.company} | {e.position} | {e.years}年")
rprint(f"  总结: {resume.summary}")
rprint(f"  类型: {type(resume).__name__}")  # 证明这是真正的 Resume 对象


# ============================================================
# 阶段2：可选字段 + 路由 —— 同一结构，不同场景填不同字段
# ============================================================
# 现实：很多 API 不会用 Union 返回不同类型，而是用一个"胖模型"
# 包含所有可能字段，用 intent 字段区分当前是什么类型。
# 这比 Union 更工程化——调用方只需要处理一个类型。

class SmartReply(BaseModel):
    """智能回复 —— 同一个模型覆盖多种场景"""
    intent: Literal["weather", "calc", "chat"] = Field(description="意图分类")
    city: Optional[str] = Field(default=None, description="天气查询时填城市")
    expression: Optional[str] = Field(default=None, description="计算查询时填表达式")
    answer: str = Field(description="最终回复文本")
    confidence: float = Field(default=1.0, ge=0, le=1, description="置信度 0-1")


smart_model = model.with_structured_output(SmartReply)

queries = [
    "上海明天天气怎么样",
    "帮我算一下 123 * 456",
    "讲个笑话",
]

rprint("\n[bold]阶段2: 可选字段 + 路由[/bold]")
for q in queries:
    result = smart_model.invoke(q)
    rprint(f"  输入: {q}")
    rprint(f"  → intent={result.intent}, answer={result.answer}")
    # 根据 intent 路由到不同处理逻辑
    if result.intent == "weather":
        rprint(f"    天气查询: {result.city}, 置信度={result.confidence}")
    elif result.intent == "calc":
        rprint(f"    计算: {result.expression}, 置信度={result.confidence}")
    else:
        rprint(f"    闲聊, 置信度={result.confidence}")


# ============================================================
# 阶段3：批量提取 —— 这才是生产环境的真实用法
# ============================================================

class NewsItem(BaseModel):
    """新闻条目"""
    title: str = Field(description="新闻标题")
    category: Literal["科技", "财经", "体育", "娱乐", "其他"] = Field(description="分类")
    sentiment: Literal["正面", "负面", "中性"] = Field(description="情感倾向")
    keywords: List[str] = Field(description="关键词, 3-5个", min_length=3, max_length=5)

news_texts = [
    "苹果公司今天发布了新款iPhone，股价上涨3%，分析师普遍看好。",
    "国足在世界杯预选赛中0:3不敌对手，出线形势严峻。",
    "某明星宣布离婚，粉丝纷纷表示惋惜，话题登上热搜第一。",
]

news_model = model.with_structured_output(NewsItem)
rprint("\n[bold]阶段3: 批量提取 + 后续处理[/bold]")
for i, text in enumerate(news_texts):
    news = news_model.invoke(text)
    rprint(f"  [{i+1}] {news.title}")
    rprint(f"      分类: {news.category} | 情感: {news.sentiment}")
    rprint(f"      关键词: {news.keywords}")

    # 关键：拿到的是 Python 对象，可以直接写业务逻辑
    if news.sentiment == "负面":
        rprint(f"      [!] 负面新闻，可能需要人工审核")