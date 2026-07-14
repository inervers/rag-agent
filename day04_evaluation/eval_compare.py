"""固定管道 vs Function Calling 对比评测

指标：
  1. 语义相似度  — 回答与参考答案的余弦相似度
  2. 覆盖度       — 回答是否覆盖参考答案所有关键概念
  3. 忠实度       — 回答是否忠于检索到的上下文

流程：
  同一套知识库 → 同一组测试题 → 两套架构分别跑 → 三指标对比
"""

import sys, os, json, time

_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

os.environ.setdefault("HF_HOME", r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

os.environ.setdefault("DEEPSEEK_API_KEY", "")
if not os.environ["DEEPSEEK_API_KEY"]:
    env_path = os.path.join(os.path.dirname(__file__), "..", "day03_rag_api_fc", ".env")
    if os.path.isfile(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "DEEPSEEK_API_KEY":
                    os.environ["DEEPSEEK_API_KEY"] = v.strip()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
if not DEEPSEEK_API_KEY:
    print("需要设置 DEEPSEEK_API_KEY")
    exit(1)

import numpy as np
import httpx
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import chromadb
from chromadb.api.types import EmbeddingFunction

# =============================================
# 共享组件
# =============================================

# 嵌入模型
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

def embed_texts(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        pooled = model(**inputs).last_hidden_state.mean(dim=1)
    return (pooled / torch.norm(pooled, dim=1, keepdim=True)).numpy()

class MiniLMEmbedding(EmbeddingFunction):
    def __call__(self, texts):
        return embed_texts(texts).tolist()

# Chroma 知识库（复用 day03 的持久化数据）
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "day03_rag_api_fc", "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=chromadb.config.Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="rag_knowledge", embedding_function=MiniLMEmbedding())

# DeepSeek LLM
llm_client = httpx.Client(timeout=30)

def call_llm(messages, stream=False, tools=None):
    body = {
        "model": "deepseek-v4-flash", "messages": messages,
        "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": stream,
    }
    if tools:
        body["tools"] = tools
    r = llm_client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    if r.status_code != 200:
        print(f"  请求失败 ({r.status_code}): {r.text[:200]}")
    r.raise_for_status()
    return r.json()

def llm_ask(system: str, user: str) -> str:
    return call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])["choices"][0]["message"]["content"]

# =============================================
# 架构 A：固定管道
# =============================================

FIXED_SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个技术问答助手。基于检索到的上下文回答问题。如果上下文不够回答就说不知道。不要编造。"),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])

def build_fixed_pipeline():
    def retrieve(question):
        results = collection.query(query_texts=[question], n_results=3)
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else "无相关信息"

    def format_input(data):
        return {
            "context": data["context"],
            "question": data["question"],
        }

    def invoke_llm(data):
        prompt = FIXED_SYSTEM_PROMPT.format_messages(**data)
        role_map = {"human": "user", "system": "system"}
        msgs = [{"role": role_map.get(m.type, m.type), "content": m.content} for m in prompt]
        return call_llm(msgs)["choices"][0]["message"]["content"]

    def run(question):
        context = retrieve(question)
        return invoke_llm({"context": context, "question": question})

    return run

fixed_run = build_fixed_pipeline()

# =============================================
# 架构 B：Function Calling
# =============================================

FC_SYSTEM_PROMPT = (
    "You are an AI assistant with dedicated tools for specific tasks.\n"
    "Available tools:\n"
    "  search_knowledge : search the vector knowledge base\n"
    "  summarize        : summarize any text\n"
    "  translate        : translate text to another language\n\n"
    "Rules:\n"
    "1. FOR TECHNICAL QUESTIONS, ALWAYS use search_knowledge first.\n"
    "2. Answer in the same language as the user."
)

FC_TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "搜索知识库（向量语义检索），查找与问题相关的文档",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "对一段文本进行摘要总结",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "translate",
        "description": "将文本翻译为目标语言",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string"},
            },
            "required": ["text", "target_language"],
        },
    }},
]

def fc_search(query):
    results = collection.query(query_texts=[query], n_results=3)
    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else "未找到相关信息"

def fc_run(question):
    msgs = [
        {"role": "system", "content": FC_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(6):
        resp = call_llm(msgs, tools=FC_TOOLS)
        msg = resp["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return msg["content"]
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
        for tc in msg["tool_calls"]:
            fname = tc["function"]["name"]
            fargs = json.loads(tc["function"]["arguments"] or "{}")
            if fname == "search_knowledge":
                result = fc_search(fargs["query"])
            elif fname == "summarize":
                result = llm_ask("You are a summarizer.", f"Summarize:\n\n{fargs['text']}")
            elif fname == "translate":
                result = llm_ask(f"Translate to {fargs['target_language']}.", fargs["text"])
            else:
                result = "未知工具"
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return msgs[-1].get("content", "")

# =============================================
# 测试数据集
# =============================================

TEST_CASES = [
    {
        "question": "What is Python?",
        "reference": "Python is a high-level general-purpose programming language created by Guido van Rossum, first released in 1991. It emphasizes code readability and supports multiple programming paradigms.",
    },
    {
        "question": "What is PyTorch and who developed it?",
        "reference": "PyTorch is an open-source ML framework developed by Meta AI, released in 2016. It features dynamic computation graphs and GPU acceleration.",
    },
    {
        "question": "What is the Transformer architecture?",
        "reference": "The Transformer was introduced by Google in 2017. It uses self-attention instead of RNNs or CNNs and is the foundation for BERT, GPT, and ViT.",
    },
    {
        "question": "What is RAG and what problem does it solve?",
        "reference": "RAG combines a retriever and generator. The retriever searches a knowledge base for relevant documents, used as context for the LLM. It reduces hallucination and enables knowledge updates without retraining.",
    },
    {
        "question": "What is Chroma?",
        "reference": "Chroma is an open-source vector database for AI applications, supporting cosine similarity and persistent storage, integrating with LangChain and LlamaIndex.",
    },
]

# =============================================
# 评测指标
# =============================================

METRIC_SYSTEM = (
    "You are an evaluation assistant. Given a reference answer and a candidate answer, "
    "rate ONLY on the specified criterion. "
    "Output ONLY a number between 0 and 1, nothing else."
)

def semantic_similarity(a: str, b: str) -> float:
    """余弦相似度"""
    emb = embed_texts([a, b])
    cos_sim = np.dot(emb[0], emb[1])
    return float(round(cos_sim, 4))

def coverage(reference: str, answer: str) -> float:
    """覆盖度：参考回答的关键概念是否都在回答中出现"""
    score = llm_ask(
        METRIC_SYSTEM + "\nCriterion: COVERAGE. Does the candidate answer cover all key concepts in the reference?",
        f"Reference: {reference}\n\nCandidate: {answer}\n\nCoverage score (0-1):",
    )
    try:
        return float(score.strip())
    except:
        return 0.0

def faithfulness(answer: str) -> float:
    """忠实度：回答是否基于已知事实，没有编造"""
    score = llm_ask(
        METRIC_SYSTEM + "\nCriterion: FACTUAL ACCURACY. Is the answer factually accurate and not hallucinating?",
        f"Answer: {answer}\n\nFaithfulness score (0-1):",
    )
    try:
        return float(score.strip())
    except:
        return 0.0

# =============================================
# 评测循环
# =============================================

print("=" * 70)
print("  固定管道 vs Function Calling 对比评测")
print("=" * 70)
print(f"测试集：{len(TEST_CASES)} 题")
print()

results = []

for i, tc in enumerate(TEST_CASES):
    q = tc["question"]
    ref = tc["reference"]
    print(f"[{i+1}/{len(TEST_CASES)}] {q}")
    print(f"  参考：{ref[:60]}...")

    # 固定管道
    t0 = time.time()
    ans_a = fixed_run(q)
    t_a = round(time.time() - t0, 2)
    print(f"  固定管道 ({t_a}s): {ans_a[:80]}...")

    # FC
    t0 = time.time()
    ans_b = fc_run(q)
    t_b = round(time.time() - t0, 2)
    print(f"  FC ({t_b}s): {ans_b[:80]}...")

    # 指标
    sim = semantic_similarity(ans_a, ans_b)
    cov_a = coverage(ref, ans_a)
    cov_b = coverage(ref, ans_b)
    fai_a = faithfulness(ans_a)
    fai_b = faithfulness(ans_b)

    results.append({
        "question": q,
        "time_fixed": t_a,
        "time_fc": t_b,
        "similarity": sim,
        "coverage_fixed": cov_a,
        "coverage_fc": cov_b,
        "faithfulness_fixed": fai_a,
        "faithfulness_fc": fai_b,
    })

    print(f"  📊 语义相似度: {sim}")
    print(f"  📊 覆盖度:  固定={cov_a} FC={cov_b}")
    print(f"  📊 忠实度:  固定={fai_a} FC={fai_b}")
    print()

# =============================================
# 汇总
# =============================================

print("=" * 70)
print("  汇总")
print("=" * 70)
print()

avg_sim = np.mean([r["similarity"] for r in results])
avg_cov_fixed = np.mean([r["coverage_fixed"] for r in results])
avg_cov_fc = np.mean([r["coverage_fc"] for r in results])
avg_fai_fixed = np.mean([r["faithfulness_fixed"] for r in results])
avg_fai_fc = np.mean([r["faithfulness_fc"] for r in results])

print(f"{'指标':<20} {'固定管道':<12} {'FC':<12} {'差异':<12}")
print("-" * 56)
print(f"{'覆盖度':<20} {avg_cov_fixed:<12.4f} {avg_cov_fc:<12.4f} {avg_cov_fc - avg_cov_fixed:<+12.4f}")
print(f"{'忠实度':<20} {avg_fai_fixed:<12.4f} {avg_fai_fc:<12.4f} {avg_fai_fc - avg_fai_fixed:<+12.4f}")
print(f"{'语义相似度（间）':<20} {avg_sim:<12.4f} {'':<12} {'':<12}")
print()

# 时间对比
avg_t_fixed = np.mean([r["time_fixed"] for r in results])
avg_t_fc = np.mean([r["time_fc"] for r in results])
print(f"{'平均耗时':<20} {avg_t_fixed:<12.2f}s {avg_t_fc:<12.2f}s {avg_t_fc - avg_t_fixed:<+12.2f}s")
print()

# 逐题详情
print("  逐题详情")
print()
print(f"{'#':<3} {'语义相似度':<12} {'覆盖(固定)':<12} {'覆盖(FC)':<12} {'忠实(固定)':<12} {'忠实(FC)':<12}")
print("-" * 63)
for i, r in enumerate(results):
    print(f"{i+1:<3} {r['similarity']:<12.4f} {r['coverage_fixed']:<12.4f} {r['coverage_fc']:<12.4f} "
          f"{r['faithfulness_fixed']:<12.4f} {r['faithfulness_fc']:<12.4f}")

print()
print("评测完成")
