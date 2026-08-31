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
| `store_tools_demo.py` | 工具访问长期记忆：InjectedStore 注入 + AI 自动读写完整闭环 |
| `middleware_demo.py` | 中间件机制演示（before_model / after_model） |

### 综合项目

| 文件 | 内容 |
|------|------|
| `smart_assistant.py` | 智能助手：流式对话 + Tavily/计算工具 + 混合记忆 + 结构化输出 |

### MCP（Model Context Protocol）

| 文件 | 内容 |
|------|------|
| `mcp_demo_server.py` | MCP server：FastMCP 提供 add / current_time / web_search（Tavily 真搜索） |
| `mcp_demo_client.py` | MCP client：连接 server，`list_tools` 发现工具、`call_tool` 调用 |
| `mcp_agent_demo.py` | `langchain-mcp-adapters`：把 MCP 工具接进 `create_agent`，模型自动调用 |
| `mcp_demo.py` | 一键跑两个演示：client 手动调用 + agent 自动调用 |

## MCP 要点（学习笔记）

- **是什么**：Model Context Protocol，AI 工具接入的开放标准（工具界的 USB-C）。工具写一次，Claude / Codex / LangChain 都能用，与模型解耦
- **三角色**：Host（AI 应用，决定用哪些 server）/ Client（应用内部的连接器，一个 server 一个）/ Server（独立进程，提供 Tools / Resources / Prompts）
- **传输**：stdio（本地子进程，学习用，一对一生死与共）/ HTTP+SSE（远程常驻，生产用，可多客户端共享）
- **和 bind_tools 的区别**：bind_tools = 发给具体模型的专用格式；MCP = 标准协议，任何 MCP 宿主都能发现和调用
- **流程**：`initialize` 握手 → `list_tools` 发现 → `call_tool` 调用
- **生命周期**：stdio 下 server 是 client 的子进程，client 退出 server 随之结束；要常驻/共享需 HTTP 模式
- **版本坑**：`langchain-mcp-adapters 0.3.2` 要求 `mcp<2`，所以环境里用 1.x 的 `FastMCP`；mcp 2.x 改名为 `MCPServer`，适配器跟上后可升级

## 环境要点

- deepseek 思考模式不支持 `tool_choice` 强制结构化 → 结构化输出用 Qwen（`enable_thinking=False` + `ToolStrategy`）
- Windows 终端 GBK 编码遇到 emoji 会报错 → 脚本开头 `sys.stdout.reconfigure(encoding="utf-8")`
- pip 装包失败（连不上 PyPI）→ 用清华镜像：`pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 本地代理（127.0.0.1:7897）没开时，git 推送用直连：`git -c http.proxy= -c https.proxy= push origin master`

## 学习路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| 基础调用 | invoke / stream / batch / 异步 | 完成 |
| 消息与记忆 | messages、提示词模板 | 完成 |
| 结构化输出 | Pydantic / ToolStrategy / Union | 完成 |
| Agent | create_agent / StateGraph / 工具 | 完成 |
| 记忆系统 | checkpointer / Store / 混合记忆 | 完成 |
| 中间件 | hook 机制 / middleware | 完成 |
| RAG | 手动 RAG → 切分+评分检索 → Agent 化 RAG（rag_demo.py） | 完成 |
| MCP | server / client / create_agent 接入 | 完成 |
