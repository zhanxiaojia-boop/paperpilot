import os 
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pymilvus import MilvusClient,AnnSearchRequest,RRFRanker
load_dotenv(override=True)

client = MilvusClient(uri="http://localhost:19530")
COLLECTION = "my_collection_1"         
llm = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
             base_url=os.getenv("DASHSCOPE_BASE_URL"))

def hybrid_search(query: str, limit: int = 2):
    dense = AnnSearchRequest(data=[query], anns_field="dense", param={}, limit=3)
    sparse = AnnSearchRequest(data=[query], anns_field="sparse", param={}, limit=3)
    res = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense, sparse],
        ranker=RRFRanker(60),
        limit=limit,
        output_fields=["content", "h2"],
    )
    return res[0]

def ask(question: str) -> str:
    hits = hybrid_search(question)
    context = ""
    for h in hits:
        chapter = h["entity"].get("h2", "未知来源")
        context = context + f"【来源：{chapter}】\n{h['entity']['content']}\n\n"
    context = context.strip()
    if not context:
        return "文档中没有找到相关内容。"
    resp = llm.chat.completions.create(
        model="qwen-plus",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "只根据给定资料回答，资料里没有就说不知道，不要编造。"},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
    )
    return resp.choices[0].message.content.strip()

app = FastAPI(title="PaperPilot API")

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def api_ask(q: Query):
    return {"answer": ask(q.question)}