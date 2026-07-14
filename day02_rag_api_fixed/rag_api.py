"""L4-Day1: RAG API 服务（FastAPI + LangChain LCEL + DeepSeek）"""

import sys, os

# ── 沙箱路径修正（安装 --user 的包） ──
_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HOME", r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import numpy as np
import httpx
from transformers import AutoTokenizer, AutoModel
import torch

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =============================================
# 0. 模型（DeepSeek via httpx）
# =============================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("请设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

class ChatDeepSeek:
    def __init__(self, model="deepseek-v4-flash", temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.client = httpx.Client(timeout=30)

    def invoke(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        resp = self.client.post(
            "https://api.deepseek.com/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

llm = ChatDeepSeek()

# =============================================
# 1. 嵌入（手写 MiniLM，LangChain HuggingFaceEmbeddings
#    依赖 sentence-transformers，装完后可换回官方）
# =============================================

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    _load_model()
    inputs = _tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        pooled = _model(**inputs).last_hidden_state.mean(dim=1)
    vectors = (pooled / torch.norm(pooled, dim=1, keepdim=True)).numpy()
    return [v for v in vectors]

class HandmadeEmbeddings:
    def embed_documents(self, texts):
        return embed_texts(texts)
    def embed_query(self, text):
        return embed_texts([text])[0]

embeddings = HandmadeEmbeddings()

# =============================================
# 2. 向量存储
# =============================================

class SimpleVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []

    def add_documents(self, documents):
        texts = [d.page_content for d in documents]
        self.vectors = self.embeddings.embed_documents(texts)
        self.documents = documents

    def similarity_search(self, query, k=2):
        q_vec = self.embeddings.embed_query(query)
        scores = np.dot(self.vectors, q_vec)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]

vectorstore = SimpleVectorStore(embeddings)

# =============================================
# 3. 初始化知识库
# =============================================

raw_docs = [
    Document("Python was created by Guido van Rossum in 1991. It is a high-level programming language emphasizing readability."),
    Document("PyTorch was developed by Meta AI. It provides GPU-accelerated tensor computation and automatic differentiation."),
    Document("The Transformer architecture was introduced by Google in 2017. It uses self-attention instead of recurrence."),
    Document("RAG combines a retriever with a generator. The retriever finds relevant documents, then the generator produces an answer."),
    Document("Chroma is an open-source vector database. It stores embeddings and enables similarity search for AI applications."),
    Document("LangChain is a framework for developing LLM applications. It provides chains, agents, and retrieval strategies."),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(raw_docs)
vectorstore.add_documents(chunks)
print(f"▶ 知识库已加载：{len(raw_docs)} 篇 → {len(chunks)} 个块")

# =============================================
# 4. Prompt 模板
# =============================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise QA assistant. Answer based only on the given information."),
    ("human", "Information:\n{context}\n\nQuestion:\n{question}"),
])

# =============================================
# 5. LCEL 管道（可复用）
# =============================================

def retrieve(question):
    return vectorstore.similarity_search(question, k=2)

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

def llm_invoke(msgs):
    return llm.invoke([
        {"role": "user" if m.type == "human" else m.type, "content": m.content}
        for m in msgs
    ])

rag_chain = (
    {
        "context": RunnableLambda(retrieve) | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(lambda d: prompt.format_messages(**d))
    | RunnableLambda(llm_invoke)
)

# =============================================
# 6. FastAPI 服务
# =============================================

app = FastAPI(title="RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str
    docs_count: int

@app.get("/health", response_model=HealthResponse)
def health():
    """服务健康检查"""
    return HealthResponse(status="ok", docs_count=len(vectorstore.documents))

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """接收问题，返回 RAG 回答"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    answer = rag_chain.invoke(request.question)
    return QueryResponse(answer=answer)

# =============================================
# 入口
# =============================================

if __name__ == "__main__":
    import uvicorn
    print("启动 RAG API 服务...")
    print("  健康检查 → http://localhost:8000/health")
    print("  问答接口 → POST http://localhost:8000/query")
    uvicorn.run(app, host="0.0.0.0", port=8000)
