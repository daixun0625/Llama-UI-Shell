#!/usr/bin/env python3
# e2e_stop_hang.py — 验证 35B 挂起场景停止链路（无头，独立进程运行，便于外部 timeout 兜底）
import tkinter as tk, subprocess, time, sys, queue
sys.path.insert(0, "/Users/dx/llama-shell")
import gui
from gui import App

root = tk.Tk()
app = App(root)
app.port_var.set("8081")

HANG = '''
import signal, socket, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
import sys; sys.argv
PHASE = sys.argv[1] if len(sys.argv) > 1 else "B"
if PHASE == "B":
    time.sleep(2.0)  # 模拟 35B 加载中（还没 listen）
sk = socket.socket(); sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sk.bind(("127.0.0.1", 8081)); sk.listen(1)
time.sleep(120)
'''

def lsof_pids(port):
    return subprocess.run(["lsof","-ti",f":{port}"],capture_output=True,text=True,timeout=5).stdout.split()

def drain_q():
    out=[]
    while True:
        try: out.append(app.q.get_nowait())
        except queue.Empty: break
    return out

# ===== B 场景：忽略 SIGTERM 且 _stop 时尚未 listen（最恶劣）=====
pB = subprocess.Popen(["python3","-c",HANG,"B"], start_new_session=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
app.proc = pB
time.sleep(1.2)
assert not lsof_pids(8081), "前提不成立：B 阶段尚未 listen"
t0=time.time(); app._stop(); t1=time.time()
print(f"[B] _stop 返回 {t1-t0:.3f}s → {'非阻塞 ✓' if t1-t0<1.0 else '卡死 ✗'}", flush=True)
time.sleep(6.5)   # _reap: 5s 等待 + SIGKILL + kill_port
left = lsof_pids(8081)
print(f"[B] 挂起进程 {pB.pid} poll={pB.poll()} (非 0=已杀) | 端口残留: {left if left else '无 ✓'}", flush=True)
for m in drain_q():
    if "[GUI]" in m: print("   日志:", m.replace("OUT:","").strip()[:70], flush=True)

# ===== A 场景：立即 listen + 忽略 SIGTERM → _stop 的 kill_port 秒清 =====
time.sleep(0.5)
pA = subprocess.Popen(["python3","-c",HANG,"A"], start_new_session=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
app.proc = pA
time.sleep(1.2)
assert lsof_pids(8081), "前提不成立：A 应已 listen"
t0=time.time(); app._stop(); t1=time.time()
print(f"[A] _stop 返回 {t1-t0:.3f}s", flush=True)
time.sleep(1.5)
left = lsof_pids(8081)
print(f"[A] 监听进程 {pA.pid} poll={pA.poll()} (非 0=已杀) | 端口残留: {left if left else '无 ✓'}", flush=True)

time.sleep(5.0)  # B 的 reap 线程收尾（其 kill_port 为空操作）
alive = [p for p in (pA.pid, pB.pid) if subprocess.run(["kill","-0",str(p)],capture_output=True,timeout=5).returncode==0]
print(f"全部测试进程已退出: {'✓' if not alive else '✗ 残留 '+str(alive)}", flush=True)

root.destroy()
import os
os._exit(0)  # 测试环境强制退出（App 的扫描线程可能为非 daemon，不影响结论）
