#!/usr/bin/env python3
"""e2e: 默认档位 + 按模型持久化（真实 Tk 主线程）"""
import sys, os, json
sys.path.insert(0, "/Users/dx/llama-shell")
import tkinter as tk
import gui as g

if os.path.exists(g.STATE_FILE):
    os.remove(g.STATE_FILE)

root = tk.Tk()
app = g.App(root)

# --- 1) 新默认值 ---
deflt = dict(ctx="65536", temp="0.75", repeat_penalty="1.1",
             presence_penalty="0.05", cache_type_k="q8_0", cache_type_v="q8_0", flash_attn="on")
for k, v in deflt.items():
    got = app.pv[k].get()
    assert got == v, f"default {k}: {got!r} != {v!r}"
print("PASS 1: 新默认档位 (ctx=65536, temp=0.75, rp=1.1, pp=0.05, kv=q8_0, fa=on)")

# --- 2) 模型列表 + 选 35B MoE ---
models = g.scan_models()
assert models, "scan_models 返回空"
app._fill_tree(models)
i35 = next(i for i, m in enumerate(models) if "35B" in m.get("name", ""))
iid = app.tree.get_children()[i35]
app.tree.selection_set(iid)
app.tree.event_generate("<<TreeviewSelect>>")
st = app.selected
assert st and "35B" in st["name"], st
# 无历史状态 → 保持默认
assert app.pv["ctx"].get() == "65536"
# 默认参数下命令应含关键旗标
cmd = app.cmd.get("1.0", "end").strip()
for tok in ("65536", "0.75", "q8_0"):
    assert tok in cmd, f"cmd 缺 {tok}:\n{cmd}"
print("PASS 2: 选 35B（无历史）保持默认；命令含 65536/0.75/q8_0")

# --- 3) 模拟用户编辑 35B → 应自动保存 ---
app.pv["ctx"].set("32768")
app.pv["top_p"].set("0.90")
app.ovr.config(state="normal")
app.ovr.delete("1.0", "end")
app.ovr.insert("1.0", "--no-mmap on")
app.engine_var.set("vanilla")
data = json.load(open(g.STATE_FILE))
e = data[st["path"]]
assert e["params"]["ctx"] == "32768", e
assert e["params"]["top_p"] == "0.90"
assert e["override"] == "--no-mmap on"
assert e["engine"] == "vanilla"
print("PASS 3: 编辑 35B 面板/覆盖/引擎 → 自动落盘 gui_state.json")

# --- 4) 模拟重启：抹掉面板（抑制保存），切走再切回 35B → 应恢复 ---
app._restoring = True
app.pv["ctx"].set("16384")
app.pv["top_p"].set("0.95")
app.ovr.delete("1.0", "end")
app.ovr.insert("1.0", "")
app._restoring = False
# 切到别的模型（dense 27B）
# 切到别的模型（任取一个非 35B）
i27 = next(i for i, m in enumerate(models) if i != i35)
iid27 = app.tree.get_children()[i27]
app.tree.selection_set(iid27)
app.tree.event_generate("<<TreeviewSelect>>")
assert app.selected["name"] != st["name"]
st_other = app.selected
# 再切回 35B
app.tree.selection_set(iid)
app.tree.event_generate("<<TreeviewSelect>>")
assert app.pv["ctx"].get() == "32768", app.pv["ctx"].get()
assert app.pv["top_p"].get() == "0.90"
assert app.ovr.get("1.0", "end").strip() == "--no-mmap on"
assert app.engine_var.get() == "vanilla"
# 恢复期间不应覆盖已存状态
data2 = json.load(open(g.STATE_FILE))
assert data2[st["path"]]["params"]["ctx"] == "32768"
print("PASS 4: 切走→切回 35B，参数/覆盖/引擎全部恢复；保存未被恢复过程污染")

# --- 5) 切 27B（无历史）：无条目污染；面板被“默认+推荐预填”覆盖（不再沿用上一模型的值）---
e27 = data2.get(st_other["path"])
assert e27 in (None, {}), e27
import recommend as _rec
exp_pre = _rec.prefill(st_other, app.hw)
app.tree.selection_set(iid27)
app.tree.event_generate("<<TreeviewSelect>>")
assert app.selected is st_other
for k, v in exp_pre.items():
    assert app.pv[k].get() == str(v), f"prefill 未写入 {k}: 期望 {v!r} 实际 {app.pv[k].get()!r}"
cmd2 = app.cmd.get("1.0", "end").strip()
assert cmd2 and st_other["path"] in cmd2, (cmd2[:200], st_other["path"])
print(f"PASS 5: 切 {st_other['name']}（无历史）无条目污染；预填 { {k: exp_pre[k] for k in ('ngl','ctx','cache_type','threads') if k in exp_pre} }；命令正常")

root.destroy()
print("ALL PASS")
