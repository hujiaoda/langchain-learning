# langchain-learning

学习 LangChain 的代码仓库，从模型调用到 Agent、结构化输出、记忆系统、中间件，持续更新。

## 快速开始

1. 创建 `.env` 文件，填入你的 API Key：
   ```
   DEEPSEEK_API_KEY=sk-xxx
   QWEN_API_KEY=sk-xxx
   TAVILY_API_KEY=tvly-xxx
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 文件清单

### 基础调用

| 文件 | 内容 |
|------|------|
| `init.py` | 模型初始化与 `invoke` 基础调用 |
| `stream.py` | `create_agent` 流式输出 + 多轮记忆 + 记忆截断 |
| `batch.py` | `batch` 批量调用 |
| `ainvoke.py` | asyncio 异步调用 |
| `profile.py` | LangSmith 追踪配置 |
| `ChatPromptTemplate.py` | 提示词模板 |

### 结构化输出

| 文件 | 内容 |
|------|------|
| `output.py` | `ToolStrategy` + Union 多类型输出 + 兜底类型（Qwen 环境） |
| `output-adv.py` | 嵌套结构 / 可选字段路由 / 批量提取 三阶段 |

### Agent

| 文件 | 内容 |
|------|------|
| `tool.py` | `@tool` 工具绑定 + tool_calls |
| `agent.py` | `create_agent` + Tavily 搜索 |
| `agent demo.py` | 手动循环 → create_agent → 手写 StateGraph 三阶段 |

### 记忆与中间件

| 文件 | 内容 |
|------|------|
| `checkpointer.py` | 短期记忆（checkpointer + thread_id）与长期记忆（Store）示例 |
| `middleware_demo.py` | 中间件机制演示（before_model / after_model） |

### 综合项目

| 文件 | 内容 |
|------|------|
| `smart_assistant.py` | 智能助手：流式对话 + Tavily/计算工具 + 混合记忆 + 结构化输出 |

## 环境要点

- deepseek 思考模式不支持 `tool_choice` 强制结构化 → 结构化输出用 Qwen（`enable_thinking=False` + `ToolStrategy`）
- Windows 终端 GBK 编码遇到 emoji 会报错 → 脚本开头 `sys.stdout.reconfigure(encoding="utf-8")`

## 学习路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| 基础调用 | invoke / stream / batch / 异步 | 完成 |
| 消息与记忆 | messages、提示词模板 | 完成 |
| 结构化输出 | Pydantic / ToolStrategy / Union | 完成 |
| Agent | create_agent / StateGraph / 工具 | 完成 |
| 记忆系统 | checkpointer / Store / 混合记忆 | 完成 |
| 中间件 | hook 机制 / middleware | 完成 |
| RAG | 向量检索、知识库 | 待学 |