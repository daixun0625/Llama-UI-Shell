#!/usr/bin/env python3
"""e2e: 配置另存（预设 存/载/删）+ 导出/导入 往返（monkeypatch 绕过弹窗，走真实代码路径）"""
import sys, os, json, tempfile
sys.path.insert(0, "/Users/dx/llama-shell")
import tkinter as tk
import gui as g

if os.path.exists(g.PRESET_FILE):
    os.remove(g.PRESET_FILE)

root = tk.Tk()
app = g.App(root)

# 选 35B MoE，改几个值制造"非默认"状态
models = g.scan_models()
app._fill_tree(models)
i35 = next(i for i, m in enumerate(models) if "35B" in m.get("name", ""))
iid = app.tree.get_children()[i35]
app.tree.selection_set(iid)
app.tree.event_generate("<<TreeviewSelect>>")
app.pv["ctx"].set("32768")
app.pv["temp"].set("0.6")
app.port_var.set("8123")
app.host_var.set("0.0.0.0")
app.engine_var.set("tosh")

# --- 1) _current_state 捕获 ---
st = app._current_state()
assert st["params"]["ctx"] == "32768" and st["params"]["temp"] == "0.6", st["params"]
assert st["port"] == "8123" and st["host"] == "0.0.0.0" and st["engine"] == "tosh", st
assert "override" in st and "dynamic_moe" in st
print("PASS 1: _current_state 捕获 params/override/engine/port/host/dynamic_moe")

# --- 2) 另存预设（monkeypatch askstring） ---
g.simpledialog.askstring = lambda *a, **k: "35B-高速"
app._preset_save_as()
presets = app._load_presets()
assert "35B-高速" in presets and presets["35B-高速"]["params"]["ctx"] == "32768", presets
assert os.path.exists(g.PRESET_FILE)
# 下拉应含该预设
assert "35B-高速" in app.preset_combo["values"], app.preset_combo["values"]
print("PASS 2: 另存预设 → gui_presets.json，下拉刷新")

# --- 3) 改回别的值 → 加载预设（set combo 后 _preset_load） ---
app.pv["ctx"].set("65536")
app.port_var.set("9999")
app.preset_var.set("35B-高速")
app._preset_load()
assert app.pv["ctx"].get() == "32768" and app.port_var.get() == "8123", (app.pv["ctx"].get(), app.port_var.get())
cmd = app.cmd.get("1.0", "end").strip()
assert "32768" in cmd and "8123" in cmd and "0.0.0.0" in cmd, cmd
print("PASS 3: 加载预设 复原面板 + 刷新命令（含 32768/8123/0.0.0.0）")

# --- 4) 导出（monkeypatch asksaveasfilename）→ 校验 JSON 结构 ---
tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
g.filedialog.asksaveasfilename = lambda *a, **k: tmp
app._export()
data = json.load(open(tmp, encoding="utf-8"))
assert data["app"] == "llama-shell" and "exported_at" in data
assert data["state"]["params"]["ctx"] == "32768" and data["state"]["port"] == "8123", data["state"]
print("PASS 4: 导出 JSON（封装含 state，mmproj/引擎/端口齐全）")

# --- 5) 破坏值 → 导入（monkeypatch askopenfilename）复原 ---
app.pv["ctx"].set("4096")
app.port_var.set("2000")
g.filedialog.askopenfilename = lambda *a, **k: tmp
app._import()
assert app.pv["ctx"].get() == "32768" and app.port_var.get() == "8123", (app.pv["ctx"].get(), app.port_var.get())
print("PASS 5: 导入文件 复原面板（兼容封装 state）")

# --- 6) 导入裸状态 dict（无封装）也要能解析 ---
bare = json.load(open(tmp, encoding="utf-8"))["state"]
bare2 = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
json.dump(bare, open(bare2.name, "w", encoding="utf-8"))
g.filedialog.askopenfilename = lambda *a, **k: bare2.name
app.pv["ctx"].set("8192")
app._import()
assert app.pv["ctx"].get() == "32768", app.pv["ctx"].get()
print("PASS 6: 导入裸状态 dict（向后兼容）")

# --- 7) 删除预设 ---
app.preset_var.set("35B-高速")
app._preset_delete()
assert "35B-高速" not in app._load_presets()
assert "35B-高速" not in app.preset_combo["values"]
print("PASS 7: 删除预设，下拉同步移除")

# 清理临时文件
for p in (tmp, bare2.name):
    try:
        os.remove(p)
    except OSError:
        pass
os.remove(g.PRESET_FILE)
print("\nALL 7 PASS — 配置另存(预设) + 导出/导入 往返验证通过")
