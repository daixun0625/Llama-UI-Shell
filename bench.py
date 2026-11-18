#!/usr/bin/env python3
"""对指定端口计时生成 48 token，报告 t/s。多次取中位。"""
import time, json, urllib.request, sys

PROMPT = "Explain, in a few sentences, how a modern CPU pipeline executes a stream of instructions."
N = 48
WARM = 2

def one(port: int) -> float:
    body = {"prompt": PROMPT, "max_tokens": N, "stream": False, "temperature": 0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.time() - t0
    ntok = d.get("n_tokens_predicted", N)
    return ntok / dt

if __name__ == "__main__":
    port = int(sys.argv[1])
    for _ in range(WARM):
        one(port)  # warm: 预热 KV + 线程绑定
    runs = [one(port) for _ in range(3)]
    runs.sort()
    print(f"port={port} t/s = " + " ".join(f"{x:.1f}" for x in runs) +
          f"  (median {runs[len(runs)//2]:.1f})")
