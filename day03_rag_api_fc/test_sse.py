"""SSE 流式测试客户端"""
import httpx, json, sys

question = sys.argv[1] if len(sys.argv) > 1 else "What is RAG? Summarize in Chinese"

with httpx.Client(timeout=60) as client:
    with client.stream("POST", "http://localhost:8000/query/stream",
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
