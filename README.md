# PaperPilot - RAG 文档问答系统

基于 RAG 架构的 PDF 文档问答助手，支持页码级来源溯源。

## 技术栈
- 文档解析：MinerU API / PyPDFLoader
- 向量库：Milvus（混合检索：向量 + 关键词，RRF 融合排序）
- Agent 编排：LangGraph（工具调用）
- 服务化：FastAPI，/ask 接口 + Swagger 文档
- 大模型：通义千问 qwen-plus

## 项目结构
- `Paperpilot.ipynb`：文档解析、切分、入库
- `api.py`：FastAPI 服务，混合检索 + 问答接口

## 运行
1. 复制 `.env.example` 为 `.env`，填入 API Key
2. 启动 Milvus：`docker compose up -d`
3. 启动服务：`uvicorn api:app --reload --port 8000`
4. POST `/ask`，参数 `{"question": "你的问题"}`

## 效果
问答返回答案 + 来源章节，降低大模型幻觉。
