"""启动服务 → 测试流式 → 停止"""
import sys, os, subprocess, time, httpx, json

# 启动服务
env = os.environ.copy()
env["DEEPSEEK_API_KEY"] = "sk-752b8485cf304758a53ba8ae1c49837d"
script = os.path.join(os.path.dirname(__file__), "rag_api.py")

proc = subprocess.Popen(
    [sys.executable, script],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=env, cwd=os.path.dirname(__file__),
)
time.sleep(20)

# 测试流式
def test(question):
    print(f"\n>>> {question}")
    with httpx.Client(timeout=60) as c:
        with c.stream("POST", "http://localhost:8000/query/stream",
                      json={"question": question}) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                evt = json.loads(data)
                t = evt["type"]
                if t == "token":
                    print(evt["content"], end="", flush=True)
                elif t == "tool":
                    print(f"\n[TOOL] {evt['name']}({json.dumps(evt['args'], ensure_ascii=False)})")
                elif t == "done":
                    print()
                    break

test("hi")
test("What is LangChain?")
test("What is RAG? Summarize in Chinese")

proc.terminate()
proc.wait()
print("\nDone.")
