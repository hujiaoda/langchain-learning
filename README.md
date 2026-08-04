# langchain-learning

学习 LangChain 的代码仓库，从模型调用到 RAG、Agent，持续更新。

## 目录结构

```
├── config.py          # 公共配置（API Key、Base URL）
├── init.py            # 基础：模型初始化与 invoke 调用
├── stream.py          # 基础：流式输出 + 多轮对话记忆
├── requirements.txt   # 依赖
│
├── 02-rag/            # RAG 检索增强生成（待学）
├── 03-agents/         # Agent 工具调用（待学）
└── ...
```

## 快速开始

1. 创建 `.env` 文件，填入你的 API Key：
   ```
   DEEPSEEK_API_KEY=sk-xxx
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 运行示例：
   ```bash
   python init.py    # 基础调用
   python stream.py  # 流式多轮对话
   ```

## 学习路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| 基础调用 | invoke / stream / batch / 异步调用 | 完成 |
| 消息与记忆 | messages 容器、多轮对话 | 完成 |
| RAG | 向量数据库、检索增强生成 | 待学 |
| Agent | 工具调用、多步推理 | 待学 |