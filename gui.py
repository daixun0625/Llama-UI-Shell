#!/usr/bin/env python3
"""llama-shell GUI — 轻量原生 Tkinter。

功能：
  * 模型库浏览（/Volumes/CT-nvme/.../models 下 .gguf，后台解析，不卡 UI）
  * 选中模型 → 自动推荐推理参数（移植 ToshLLM 启发式，recommend.py）
  * 双引擎：vanilla（自动探测 bin/llama-b* 最高编号）/ ToshLLM（自动或手动选；MoE 自动走 tosh）
  * 生成后端命令、一键运行、实时日志、/health 就绪指示、测速
  * 停止

零第三方 GUI 依赖（仅标准库 tkinter + 本目录 recommend/gguf_fast）。
运行：python3 gui.py
"""
import os, re, sys, glob, time, queue, threading, subprocess, signal, json, urllib.request, psutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import gguf_fast, recommend  # noqa: E402

MODELS_DIR_DEFAULT = "/Volumes/CT-nvme/llama-bin-win-vulkan-x64/models"
TOSH_BIN_DEFAULT = "/Applications/ToshLLM.app/Contents/Resources/bin/llama-server"
# 系统默认 vanilla llama-server 探测链：
# ① 系统 PATH（which llama-server，conda/brew/源码安装都受益）
# ② 本地兜底 bin/llama-b*/llama-server 中编号最高者（PATH 里没有才用；
#    升级 llama 只需在 bin/ 放新目录，自动生效，不锁定旧版本）
VANILLA_BIN_FALLBACK = recommend.latest_vanilla_bin()
def _find_default_llama_server() -> str:
    import shutil
    p = shutil.which("llama-server")
    if p:
        return p
    return VANILLA_BIN_FALLBACK
VANILLA_BIN_DEFAULT = _find_default_llama_server()
SKIP = ("mmproj", "clip", "whisper", "mtp")   # mtp=独立 MTP draft 文件，不进主模型库

# 配置文件（模型库路径、引擎路径等）
import json as _json
CONFIG_FILE = os.path.join(BASE, "llama-shell-config.json")

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}

def _save_config(cfg: dict):
    try:
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_FILE)
    except Exception:
        pass

_cfg = _load_config()
MODELS_DIR = _cfg.get("models_dir", MODELS_DIR_DEFAULT)
VANILLA_ABS = _cfg.get("vanilla_bin", os.path.abspath(VANILLA_BIN_DEFAULT))
TOSH_ABS = _cfg.get("tosh_bin", TOSH_BIN_DEFAULT)
STATE_FILE = os.path.join(BASE, "gui_state.json")   # 按模型持久化参数面板/覆盖/引擎
PRESET_FILE = os.path.join(BASE, "gui_presets.json")  # 命名预设（可跨模型复用）

# 允许推荐计划写入"下拉表之外的具体数字"的项 → 用 Combobox（可下拉 + 手动输入）
FREE_EDIT_KEYS = {"ngl", "threads", "spec_draft_model", "spec_draft_n_max", "spec_type", "mmproj", "mmproj_device"}

def _build_mtp_cands(models_dir: str) -> tuple:
    """返回 (MTP_CANDS, MTP_DISPLAY, MTP_MAP)。
    MTP 独立 draft 文件自动识别（同视觉塔逻辑）：文件名含 'mtp'（大小写不敏感）即命中，
    覆盖 mtpdraft-* / *-MTP* / *-mtp.gguf 等命名；SKIP 中的附属文件除外。
    显示名截断 30 字符，MTP_MAP 显示名→全路径；default/off → 空串（省略 --spec-draft-model）。
    目录不存在时返回空候选，不抛异常。"""
    empty_map = {"default": "", "off": ""}
    if not os.path.isdir(models_dir):
        return [], ["off"], empty_map
    cands = sorted(os.path.join(models_dir, b) for b in os.listdir(models_dir)
                   if b.lower().endswith(".gguf") and not b.startswith(".")
                   and "mtp" in b.lower()
                   and not any(s in b.lower() for s in SKIP if s != "mtp"))
    m_map = {os.path.basename(p)[:30]: p for p in cands}
    m_map["default"] = ""
    m_map["off"] = ""
    return cands, ["off"] + [os.path.basename(p)[:30] for p in cands], m_map

def _build_mmproj_cands(models_dir: str) -> tuple:
    """返回 (MMPROJ_CANDS, MMPROJ_DISPLAY, MMPROJ_MAP)"""
    cands = sorted(glob.glob(os.path.join(models_dir, "mmproj*.gguf")))
    display = ["off"] + [os.path.basename(p)[:30] for p in cands]
    m_map = {os.path.basename(p)[:30]: p for p in cands}
    m_map["off"] = ""
    m_map["auto"] = ""
    return cands, display, m_map

# 模块级初始候选（基于配置目录），GUI 切换目录后动态更新
MTP_CANDS, MTP_DISPLAY, MTP_MAP = _build_mtp_cands(MODELS_DIR)
MMPROJ_CANDS, MMPROJ_DISPLAY, MMPROJ_MAP = _build_mmproj_cands(MODELS_DIR)

def binary_for(engine: str) -> str:
    return TOSH_ABS if engine == "tosh" else VANILLA_ABS


def health(port: int, host: str = "127.0.0.1") -> bool:
    try:
        return urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1).status == 200
    except Exception:
        return False


def kill_port(port: int) -> list:
    """清理占用指定端口的所有进程（lsof -ti :port → kill -9），返回被杀的 PID 列表。
    11-13 修复：lsof 必须带冒号（":8080"=按端口查；"8080"=按 PID 查→永远空）——
    旧写法让本兜底从上线起就没真正杀过任何孤儿，是 35B 停不掉/内存残留的深根因。"""
    import subprocess
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"],
                                      text=True, timeout=3, stderr=subprocess.DEVNULL)
        pids = [int(x) for x in out.split() if x.strip()]
        if pids:
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", str(pid)],
                                   timeout=2, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            return pids
    except Exception:
        pass
    return []


def bench_once(port: int, host: str = "127.0.0.1", n: int = 48) -> float:
    """对运行中的 server 生成 n token，返回 t/s。"""
    body = {"prompt": "Explain in a few sentences how a CPU pipeline works.",
            "max_tokens": n, "stream": False, "temperature": 0}
    req = urllib.request.Request(f"http://{host}:{port}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return d.get("n_tokens_predicted", n) / (time.time() - t0)


def scan_models(models_dir: str = None) -> list:
    """后台线程用：解析所有主模型头，返回 st 列表。"""
    if models_dir is None:
        models_dir = MODELS_DIR
    out = []
    for p in sorted(glob.glob(os.path.join(models_dir, "*.gguf"))):
        b = os.path.basename(p)
        if any(s in b.lower() for s in SKIP):
            continue
        try:
            h = gguf_fast.parse_gguf_header(p)
            st = recommend.model_stats(h)
        except Exception as e:
            st = {"name": b, "path": p, "file_gb": 0, "is_moe": False,
                  "total_b": 0, "active_b": 0, "expert_count": 0,
                  "error": str(e)}
        out.append(st)
    return out


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: "queue.Queue[str]" = queue.Queue()
        self.proc = None
        self._reader_stop = False
        self._health_stop = False
        self._dynamic_moe_used = False
        self._auto_retried = False
        self._last_cmd = ""
        self.hw = recommend.detect_hardware()
        self.st_cache = {}      # path -> st
        self.selected = None    # 当前 st
        self._restoring = False # 恢复模型参数期间抑制保存
        self._mem_total = None  # sysctl hw.memsize 缓存（硬件负载面板用）
        self.models_dir = MODELS_DIR  # 当前模型库路径（可配置）
        self._closing = False         # 关闭窗口时置 True，阻止 after 回调再入主循环
        self._poll_id = None         # 80ms 日志轮询定时器句柄
        self._hw_tick_id = None      # 2s 硬件负载定时器句柄
        self.widgets = {}       # key -> 控件（Combobox/OptionMenu，用于动态更新候选值）
        self.param_defaults = {}  # key -> 全局默认值
        self._build()
        self._refresh_hw_bar()
        self._refresh_engbar()   # 启动时把配置里的引擎路径回显到 engbar（含「自动探测」提示）
        self._scan_async()
        self._poll_id    = self.root.after(80,   self._poll)
        self._hw_tick_id = self.root.after(2000, self._hw_tick)
        # 11-13 自愈：GUI 曾被强杀/停止失败时，35B 服务成孤儿（ppid=1）占端口+内存，
        # 导致下次启动 bind 8080 失败→"加载失败"。启动即清场；退出（含异常）atexit 兜底。
        self._sweep_orphans()
        import atexit
        atexit.register(self._atexit_stop)

    def _sweep_orphans(self):
        """启动自愈：清掉上次会话遗留、正占用当前端口的孤儿进程（13.5GB 级 35B 不再残留）。"""
        try:
            port = int(self.port_var.get() or 8080)
        except Exception:
            port = 8080
        self._last_port = port
        pids = kill_port(port)
        if pids:
            self._log(f"启动自愈：已清理占用端口 {port} 的残留进程 PIDs {pids}（上次会话孤儿，内存不再残留）")

    def _atexit_stop(self):
        """解释器退出时（正常关窗 / 未捕获异常路径）兜底杀进程组 + 清端口。
        GUI 若被 SIGKILL 则本钩子不执行——由下次启动的 _sweep_orphans 接力清场。"""
        try:
            proc = getattr(self, "proc", None)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                for _ in range(25):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.2)
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            try:
                kill_port(getattr(self, "_last_port", 8080))
            except Exception:
                pass
        except Exception:
            pass

    # ---------------- UI ----------------
    def _build(self):
        root = self.root
        root.title("llama-shell · 模型库 + 参数推荐 + 双引擎")
        # 初始显示框不硬编码：_build 末尾按内容需求尺寸收紧，
        # 使底边框=「后端命令」框下沿再下移 6 行小字（Menlo 11，见 _fit_frame）、右边框紧贴 ③ 参数窗右沿（见 _fit_frame）
        # 窗口关闭协议：先清理端口/状态，再销毁（防 macOS Tk 死锁）
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 顶部硬件条（raw tk.Label 在 macOS Aqua 下自定义 bg 会把 fg 变白→白底白字，
        # 故显式给深色 fg + 加深一档 bg，保证可读）
        topbar = ttk.Frame(root)
        topbar.pack(fill="x")
        self.hw_lbl = tk.Label(topbar, anchor="w", padx=10, pady=4,
                               bg="#d7e0ee", fg="#16202e",
                               font=("Menlo", 12), relief="flat")
        self.hw_lbl.pack(side="left", fill="x", expand=True)
        # 运行状态指示（READY / 运行中… / 已停止）—— _poll 的 HEALTH 事件刷新它
        self.status = tk.Label(topbar, anchor="e", padx=10, pady=4,
                                bg="#d7e0ee", fg="#16202e",
                                font=("Menlo", 12), relief="flat")
        self.status.config(text="就绪")
        self.status.pack(side="right", fill="x", expand=False)
        topbar.pack(fill="x")

        # 引擎路径配置条
        engbar = ttk.Frame(root)
        engbar.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(engbar, text="vanilla:", width=8, anchor="w").pack(side="left")
        self.vanilla_var = tk.StringVar(value=VANILLA_ABS)
        ttk.Entry(engbar, textvariable=self.vanilla_var).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(engbar, text="…", width=3,
                   command=lambda: self._browse_engine("vanilla")).pack(side="left", padx=(0, 6))
        ttk.Label(engbar, text="tosh:", width=8, anchor="w").pack(side="left")
        self.tosh_var = tk.StringVar(value=TOSH_ABS)
        ttk.Entry(engbar, textvariable=self.tosh_var).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(engbar, text="…", width=3,
                   command=lambda: self._browse_engine("tosh")).pack(side="left", padx=(0, 6))
        ttk.Button(engbar, text="应用", width=5,
                   command=self._apply_engbar).pack(side="left")

        # 模型库路径配置条
        dirbar = ttk.Frame(root)
        dirbar.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(dirbar, text="模型库:", width=8, anchor="w").pack(side="left")
        self.dir_var = tk.StringVar(value=self.models_dir)
        dir_entry = ttk.Entry(dirbar, textvariable=self.dir_var)
        dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(dirbar, text="浏览…", width=8,
                   command=self._browse_models_dir).pack(side="left")

        # 主区域：grid 两列，左列固定最小宽度，右列自适应
        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True, padx=6, pady=2)
        # 左列 540（原 600）：①②③ 参数区合计需 852px，600 会把 ③ 压到 217px
        # 致下拉截断；540 时右栏≈876 ≥ 852，三区完整显示。树列合计 470+滚动条，540 够。
        mid.grid_columnconfigure(0, minsize=540)
        mid.grid_columnconfigure(1, weight=1)

        # 左列：模型库 + 日志
        leftcol = ttk.Frame(mid, width=540)
        leftcol.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        leftcol.pack_propagate(False)

        left = ttk.LabelFrame(leftcol, text=" 模型库列表 ")
        left.pack(side="top", fill="both", expand=True)
        cols = ("name", "b", "kind", "gb", "layer")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=10)
        self._heading_base = {"name": "模型", "b": "B", "kind": "类型", "gb": "GB", "layer": "层"}
        for c, t, w in (("name", "模型", 260), ("b", "B", 44),
                        ("kind", "类型", 70), ("gb", "GB", 48), ("layer", "层", 48)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        # 排序：点击「模型 / B / 类型 / GB」表头切换升/降序
        self._sort_col = None
        self._sort_reverse = False
        self._sort_vals = {}
        for c in ("name", "kind", "b", "gb"):
            self.tree.heading(c, text=self._heading_base[c],
                              command=lambda _c=c: self._on_heading_click(_c))
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        # 日志：最底部固定
        logf = ttk.LabelFrame(leftcol, text=" 日志 ")
        logf.pack(side="bottom", fill="both", pady=(4, 0))
        log_body = ttk.Frame(logf)
        log_body.pack(fill="both", expand=True, padx=4, pady=4)
        self.log = tk.Text(log_body, height=32, state="disabled",
                           font=("Menlo", 11), bg="#0e1116", fg="#d7e0ec")
        log_sb = ttk.Scrollbar(log_body, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")

        # 右列
        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew", padx=4)

        # ① 引擎行
        f_eng = ttk.Frame(right)
        f_eng.pack(fill="x")
        ttk.Label(f_eng, text="引擎").pack(side="left", padx=(0, 4))
        self.engine_var = tk.StringVar(value="auto")
        self.engine_var.trace_add("write", lambda *_: self._save_state_entry())
        for v in ("auto", "vanilla", "tosh"):
            ttk.Radiobutton(f_eng, text=v, variable=self.engine_var,
                            value=v, command=self._refresh_recommend).pack(side="left", padx=2)
        self.dm_var = tk.BooleanVar(value=False)
        self.dm_var.trace_add("write", lambda *_: self._save_state_entry())
        ttk.Checkbutton(f_eng, text="--dynamic-moe", variable=self.dm_var,
                        command=self._refresh_recommend).pack(side="left", padx=(10, 0))
        ttk.Label(f_eng, text="端口").pack(side="left", padx=(12, 2))
        self.port_var = tk.StringVar(value="8080")
        ttk.Spinbox(f_eng, from_=1000, to=65535, textvariable=self.port_var,
                   width=7).pack(side="left")
        ttk.Label(f_eng, text="Host").pack(side="left", padx=(12, 2))
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Combobox(f_eng, textvariable=self.host_var,
                     values=["127.0.0.1", "0.0.0.0"], width=10, state="readonly").pack(side="left")
        # 工作模式：覆盖/清空（Windows spec 两种模式）
        ttk.Label(f_eng, text="模式").pack(side="left", padx=(12, 2))
        self.clear_mode_var = tk.StringVar(value="override")
        self.clear_mode_var.trace_add("write", lambda *_: self._save_state_entry())
        for v, t in (("override", "覆盖"), ("clear", "清空")):
            ttk.Radiobutton(f_eng, text=t, variable=self.clear_mode_var,
                            value=v, command=self._refresh_recommend).pack(side="left", padx=2)

        # ② 3 列参数网格（取值均已从 b12212 两个二进制 --help 实测确认）
        f_p = ttk.Frame(right)
        f_p.pack(fill="x", pady=2)
        self.pv = {}
        hw = self.hw
        tdef = str(hw.physical_cores) if hw.physical_cores else "auto"

        # 三列规格：全部首项 "default"
        specs = [
            ("① CPU/MoE", [
                ("ngl", "ngl GPU层数 -ngl",
                 ["default", "auto", "0", "4", "8", "12", "16", "24", "32", "40", "48", "64", "96", "999"], "auto"),
                ("threads", "threads -t",
                 ["default", "auto", "4", "6", "8", "12"], tdef),
                ("cpu_moe", "cpu-moe",
                 ["default", "off", "on"], "off"),
                ("moe_cache_mib", "moe-cache-mib",
                 ["default", "0", "512", "1024", "1536", "2048", "2560", "3072", "3584", "4096"], "0"),
                ("moe_cache_cpu_share", "moe-cache-cpu-share",
                 ["default", "0.0", "0.25", "0.5", "1.0"], "0.0"),
                ("batch", "batch -b",
                 ["default", "256", "512", "1024", "2048", "4096"], "512"),
                ("ubatch", "ubatch -ub",
                 ["default", "256", "512", "1024", "2048"], "512"),
                ("no_warmup", "no-warmup",
                 ["default", "auto", "off", "on"], "auto"),
                ("mmproj", "视觉塔 -mm",
                 ["default", "off"] + MMPROJ_DISPLAY[1:], "off"),
                ("mmproj_device", "--mmproj-device",
                 ["default", "none", "cpu", "gpu"], "none"),
                ("image_min_tokens", "图像最小token",
                 ["default", "256", "512", "1024", "2048"], "256"),
                ("image_max_tokens", "图像最大token",
                 ["default", "1024", "2048", "4096", "8192", "16384"], "4096"),
                ("agent", "agent",
                 ["default", "off", "on"], "on"),
            ]),
            ("② 上下文/缓存", [
                ("ctx", "ctx -c",
                 ["default", "4096", "8192", "16384", "32768", "65536", "131072"], "65536"),
                ("flash_attn", "flash-attn -fa",
                 ["default", "auto", "on", "off"], "on"),
                ("cache_type_k", "K量化 -ctk",
                 ["default", "f16", "q8_0", "q5_1", "q5_0", "q4_0", "iq4_nl"], "q8_0"),
                ("cache_type_v", "V量化 -ctv",
                 ["default", "f16", "q8_0", "q5_1", "q5_0", "q4_0", "iq4_nl"], "q8_0"),
                ("spec_type", "MTP -st",
                 ["default", "off", "draft-mtp", "draft-eagle3", "draft-simple", "ngram-simple", "ngram-cache"], "off"),
                ("spec_draft_model", "MTP draft -md",
                 ["default", "off"] + MTP_DISPLAY[1:], "default"),
                ("spec_draft_n_max", "MTP n-max",
                 ["default", "1", "2", "4", "8", "16", "32"], "default"),
                ("spec_draft_cache_type_k", "MTP draft K量化 -ctkd",
                 ["default", "f16", "q8_0", "q5_1", "q5_0", "q4_0", "iq4_nl"], "q8_0"),
                ("spec_draft_cache_type_v", "MTP draft V量化 -ctvd",
                 ["default", "f16", "q8_0", "q5_1", "q5_0", "q4_0", "iq4_nl"], "q8_0"),
                ("cache_reuse", "cache-reuse",
                 ["default", "0", "512", "2048", "4096"], "0"),
                ("cache_ram", "cache-ram -cram (MiB)",
                 ["default", "1024", "2048", "3072", "4096", "6144", "7168", "8192"], "default"),
                ("load_mode", "load-mode -lm",
                 ["default", "auto", "mmap", "mlock", "mmap+mlock", "none"], "auto"),
            ]),
            ("③ 采样/思考", [
                ("temp", "temp",
                 ["default", "0.0", "0.2", "0.5", "0.7", "0.75", "0.8", "1.0", "1.2"], "0.75"),
                ("top_k", "top-k",
                 ["default", "40", "20", "5", "100", "0"], "40"),
                ("top_p", "top-p",
                 ["default", "0.95", "0.90", "0.85", "0.99", "1.0"], "0.95"),
                ("min_p", "min-p",
                 ["default", "0.05", "0", "0.1", "0.2"], "0.05"),
                ("repeat_penalty", "repeat-penalty",
                 ["default", "1.0", "1.05", "1.1", "1.15", "1.2", "1.3"], "1.1"),
                ("presence_penalty", "presence-penalty",
                 ["default", "0", "0.05", "0.1", "0.2", "0.5", "1.0"], "0.05"),
                ("frequency_penalty", "frequency-penalty",
                 ["default", "0", "0.1", "0.2", "0.5", "1.0"], "0"),
                ("seed", "seed (-1=随机)",
                 ["default", "-1", "42", "1234"], "-1"),
                ("repeat_last_n", "repeat-last-n (惩罚窗口)",
                 ["default", "64", "128", "256", "512"], "default"),
                ("reasoning", "思考模式 reasoning",
                 ["default", "auto", "on", "off"], "auto"),
                ("reasoning_effort", "reasoning-effort",
                 ["default", "minimal", "low", "medium", "high", "max"], "default"),
                ("reasoning_format", "reasoning-format",
                 ["default", "auto", "none", "deepseek", "deepseek-legacy"], "auto"),
                ("jinja", "jinja",
                 ["default", "off", "on"], "on"),
            ]),
        ]

        for title, items in specs:
            col = ttk.LabelFrame(f_p, text=" " + title + " ")
            col.pack(side="left", fill="y", padx=2)
            col.grid_columnconfigure(1, weight=1)
            for i, (key, label, choices, default) in enumerate(items):
                var = tk.StringVar(value=default)
                self.pv[key] = var
                self.param_defaults[key] = default
                var.trace_add("write", lambda *_: self._save_state_entry())
                ttk.Label(col, text=label).grid(row=i, column=0, sticky="e", padx=(4, 2), pady=1)
                if key in FREE_EDIT_KEYS:
                    cb = ttk.Combobox(col, textvariable=var, values=choices, width=12)
                    cb.grid(row=i, column=1, sticky="e", padx=2)
                    self.widgets[key] = cb
                else:
                    # ttk.OptionMenu 吞首项：必须垫首 "default"
                    om = ttk.OptionMenu(col, var, "default", *choices)
                    om.grid(row=i, column=1, sticky="e", padx=2)
                    self.widgets[key] = om
                var.set(default)
            col._param_row_count = len(items)
        # ── REQ3 列宽统一（最终方案）：移除 ③ 区历史遗留 col2 minsize=13；
        # ①② 区 OptionMenu 'default' 候选在 macOS Aqua 主题下 reqwidth≈136px
        # （主题行为），无法在不截断的前提下缩到 ③ 区 89px；
        # 三区内部下拉框均 sticky="e" 右对齐，接受 ①② 略宽于 ③。 ──

        # 预设 / 导出 / 导入 行
        f_ps = ttk.Frame(right)
        f_ps.pack(fill="x", pady=(2, 0))
        ttk.Label(f_ps, text="预设").pack(side="left", padx=(0, 4))
        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(f_ps, textvariable=self.preset_var,
                                         width=18, state="readonly")
        self.preset_combo.pack(side="left", padx=2)
        ttk.Button(f_ps, text="加载", width=4, command=self._preset_load).pack(side="left")
        ttk.Button(f_ps, text="另存为…", width=7, command=self._preset_save_as).pack(side="left", padx=2)
        ttk.Button(f_ps, text="删除", width=4, command=self._preset_delete).pack(side="left")
        ttk.Separator(f_ps, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(f_ps, text="导出…", width=6, command=self._export).pack(side="left")
        ttk.Button(f_ps, text="导入…", width=6, command=self._import).pack(side="left", padx=2)
        self.preset_hint = ttk.Label(f_ps, text="", foreground="#888")
        self.preset_hint.pack(side="left", padx=6)
        self._refresh_preset_names()

        # ③ 自定义覆盖（用户输入的参数优先于上面面板；同名/别名自动去重）
        f_ovr = ttk.LabelFrame(right, text=" 自定义覆盖（用户参数优先；同名/别名自动去重，如 -c 覆盖 --ctx-size） ")
        f_ovr.pack(fill="x", pady=2)
        self.ovr = tk.Text(f_ovr, height=2, wrap="word", font=("Menlo", 11))
        self.ovr.pack(fill="x", padx=4, pady=2)
        self.ovr.bind("<KeyRelease>", self._on_ovr_edit)
        self.ovr_hint = ttk.Label(f_ovr, anchor="w", foreground="#666")
        self.ovr_hint.pack(fill="x", padx=6)

        # 运行/停止按钮：Canvas 自绘，钉在滚动区与硬件负载之间
        acts = ttk.Frame(right)
        acts.pack(fill="x", pady=(0, 0))

        btn_w, btn_h = 150, 42
        self.run_cv = tk.Canvas(acts, width=btn_w, height=btn_h, highlightthickness=0, bd=0)
        self.run_cv.pack(side="left", padx=(0, 4))
        self.stop_cv = tk.Canvas(acts, width=btn_w, height=btn_h, highlightthickness=0, bd=0)
        self.stop_cv.pack(side="left", padx=4)
        # ⚡生成指令（左，紧挨 运行/停止）/ 复制（右，贴窗体边）——与 运行/停止 同款 Canvas 风格
        self.gen_cv = tk.Canvas(acts, width=btn_w, height=btn_h, highlightthickness=0, bd=0)
        self.gen_cv.pack(side="left", padx=4)
        self.copy_cv = tk.Canvas(acts, width=btn_w, height=btn_h, highlightthickness=0, bd=0)
        self.copy_cv.pack(side="right", padx=(4, 0))
        # 状态机变量：_ready=server 健康就绪（运行钮变绿的门）；_cmd_available=命令已生成
        self._ready = False
        self._cmd_available = False
        # 各按钮当前标签（按下反馈用）
        self._btn_labels = {"run": "▶ 运行", "stop": "■ 停止", "gen": "⚡ 生成指令", "copy": "复制"}
        _cv_key = {self.run_cv: "run", self.stop_cv: "stop", self.gen_cv: "gen", self.copy_cv: "copy"}

        def draw_btn(cv, text, bg, fg, key=None):
            cv.delete("all")
            cv.create_rectangle(0, 0, btn_w, btn_h, fill=bg, outline="")
            cv.create_text(btn_w/2, btn_h/2, text=text, fill=fg, font=("Helvetica", 11, "bold"))
            if key:
                self._btn_labels[key] = text

        self._draw_run = lambda running, ready: draw_btn(
            self.run_cv,
            "● 就绪 · 运行中" if (running and ready) else (
                "● 正在加载" if running else
                ("● 就绪 · 点击启动" if ready else "▶ 运行（加载后启动）")),
            "#2e7d32" if ready else "#e9eaef",
            "white" if ready else "#999", key="run"
        )
        self._draw_stop = lambda running, ready: draw_btn(
            self.stop_cv,
            "■ 停止",
            "#c62828" if (running and ready) else "#d5d7dd",
            "white" if (running and ready) else "#999", key="stop"
        )
        # 生成指令 = 复制 同色（深灰 #4a5568，可用时），禁用时同为浅灰 #d5d7dd
        self._draw_gen = lambda avail: draw_btn(
            self.gen_cv, "⚡ 生成指令",
            "#4a5568" if avail else "#d5d7dd",
            "white" if avail else "#999", key="gen"
        )
        self._draw_copy = lambda avail: draw_btn(
            self.copy_cv, "复制",
            "#4a5568" if avail else "#d5d7dd",
            "white" if avail else "#999", key="copy"
        )
        # 统一重绘四个按钮（ready / running / cmd 任一状态变化时调用）

        def _paint_all():
            running = self.proc is not None and self.proc.poll() is None
            self._draw_run(running, self._ready)
            self._draw_stop(running, self._ready)
            self._draw_gen(self._cmd_available)
            self._draw_copy(self._cmd_available)
        self._refresh_btns = _paint_all

        # 视觉反馈：<ButtonPress-1> 必须 add="+"，否则顶掉 <Button-1> 点击逻辑
        # 按下：整块加深一档，文字保留原色（松手后 _refresh_btns 自动还原）
        for cv, key in _cv_key.items():
            def _press(e, c=cv, k=key):
                c.delete("all")
                c.create_rectangle(0, 0, btn_w, btn_h, fill="#c9cbd4", outline="")
                c.create_text(btn_w/2, btn_h/2, text=self._btn_labels[k], fill="#111",
                              font=("Helvetica", 11, "bold"))
            cv.bind("<ButtonPress-1>", _press, add="+")
        self.run_cv.bind("<Button-1>", lambda e: self._on_run_click(), add="+")
        self.stop_cv.bind("<Button-1>", lambda e: self._on_stop_click(), add="+")
        self.gen_cv.bind("<Button-1>", lambda e: self._on_gen_click(), add="+")
        self.copy_cv.bind("<Button-1>", lambda e: self._on_copy_click(), add="+")
        self._refresh_btns()

        # 硬件负载（实时 2s）：CPU/内存/GPU 百分比 + 条状图
        hwload = ttk.LabelFrame(right, text=" 硬件负载（实时 2s） ")
        hwload.pack(fill="x", pady=(0, 0))
        self.hwload_rows = {}
        for name in ("CPU", "内存", "GPU"):
            row = ttk.Frame(hwload)
            row.pack(fill="x", padx=4, pady=1)
            ttk.Label(row, text=name, width=5).pack(side="left")
            cv = tk.Canvas(row, height=12, highlightthickness=0, bg="#fbfcfe")
            cv.pack(side="left", fill="x", expand=True, padx=(4, 8))
            pct = ttk.Label(row, text="—", width=8, anchor="e")
            pct.pack(side="right")
            self.hwload_rows[name] = (cv, pct)

        # 后端命令框（钉最底部）
        cmdbox = ttk.LabelFrame(right, text=" 后端命令 ")
        cmdbox.pack(fill="x")
        cmd_body = ttk.Frame(cmdbox)
        cmd_body.pack(fill="x", padx=4, pady=4)
        self.cmd = tk.Text(cmd_body, height=6, wrap="word", font=("Menlo", 11))
        cmd_sb = ttk.Scrollbar(cmd_body, orient="vertical", command=self.cmd.yview)
        self.cmd.configure(yscrollcommand=cmd_sb.set)
        self.cmd.pack(side="left", fill="both", expand=True)
        cmd_sb.pack(side="right", fill="y")
        self.info = tk.Label(
            cmdbox, anchor="w", justify="left", fg="#c77700",
            font=("Menlo", 10), wraplength=560)
        self.info.pack(fill="x", padx=4, pady=(0, 4))
        self.info.config(text="（选择左侧模型以查看推荐参数）")
        # 初始框体收紧：底边框贴「后端命令」框下沿、右边框贴 ③ 参数窗右沿
        self._fit_frame()

    def _fit_frame(self):
        """把初始窗口收紧到内容需求尺寸：右沿=③参数窗右沿（最宽行），下沿=后端命令框下沿（右栏最末）。
        左栏（树/日志）自身无固有高度（fill=both，reqheight=1），整栏被右栏 reqheight 驱动；
        故取 root 的 reqwidth/reqheight 即恰好填满、无多余留白。用户后续仍可自由拉伸放大。
        底边额外下移 6 行小字（Menlo 11，11-15 主人要求）：按当前 scaling 实时测行高，
        避免硬编码像素在 Retina/不同缩放下失真。"""
        root = self.root
        root.update_idletasks()
        w, h = root.winfo_reqwidth(), root.winfo_reqheight()
        from tkinter import font as tkfont
        h += 6 * tkfont.Font(root, family="Menlo", size=11).metrics("linespace")
        if w > 400 and h > 300:  # 防异常值；否则退回 1440x954
            root.geometry(f"{w}x{h}")
        else:
            root.geometry("1440x954")

    def _refresh_hw_bar(self):
        hw = self.hw
        v_ok = os.path.exists(VANILLA_ABS)
        t_ok = os.path.exists(TOSH_ABS)
        self.hw_lbl.config(
            text=f"{hw.cpu_brand} · {hw.ram_gb:.0f} GiB · "
                 f"{hw.physical_cores}C/{hw.logical_cores}T · "
                 f"Metal 计算={'是' if hw.metal_compute else '否（显存仅显示）'}   |   "
                 f"引擎 vanilla={'✓' if v_ok else '✗缺失'}  tosh={'✓' if t_ok else '✗缺失'}")

    def _set_run_ui(self, running: bool):
        """统一重绘四个按钮：运行中→运行钮绿底“● 正在运行”、停止钮红底；
        就绪(ready)→运行钮/生成钮绿色；空闲→灰色。"""
        self._refresh_btns()

    # ---------------- 硬件负载（实时 2s） ----------------
    def _hw_tick(self):
        if getattr(self, "_closing", False):
            return
        try:
            self._update_hwload()
        except Exception:
            pass
        self._hw_tick_id = self.root.after(2000, self._hw_tick)

    def _draw_bar(self, cv, pct, note=""):
        """Canvas 条状图：灰底 + 彩色填充（<60% 蓝 / <85% 黄 / 否则红）；pct=None 时画注。"""
        cv.delete("all")
        w = max(50, cv.winfo_width())
        h = 12
        cv.create_rectangle(0, 0, w, h, fill="#e8e9ee", outline="#c7c9d1")
        if pct is None:
            cv.create_text(w / 2, h / 2, text=note or "—", fill="#888",
                           font=("Helvetica", 9))
            return
        color = ("#4a90d9" if pct < 60 else
                 "#e6a817" if pct < 85 else "#d64545")
        cv.create_rectangle(0, 0, max(2, int(w * min(100, pct) / 100)), h,
                            fill=color, outline="")

    def _update_hwload(self):
        (cv_c, pct_c), (cv_m, pct_m), (cv_g, pct_g) = (
            self.hwload_rows["CPU"], self.hwload_rows["内存"], self.hwload_rows["GPU"])
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        gpu_pct = None
        gpu_note = "无 Metal 计算设备（计算全在 CPU）"
        if self.hw.metal_compute:
            gpu_note = "powermetrics 需 sudo，未采集利用率"
        try:
            out = subprocess.run(["ioreg", "-r", "-k", "PerformanceStatistics"],
                                 capture_output=True, text=True, timeout=3).stdout
            import re
            vram_used = vram_free = gpu_util = None
            for ln in out.splitlines():
                m = re.search(r'"inUseVidMemoryBytes"\s*=\s*(\d+)', ln)
                if m:
                    vram_used = int(m.group(1))
                m = re.search(r'"vramFreeBytes"\s*=\s*(\d+)', ln)
                if m:
                    vram_free = int(m.group(1))
                m = re.search(r'"GPU Activity\(%\)"\s*=\s*(\d+)', ln)
                if m:
                    gpu_util = int(m.group(1))
            if vram_used is not None and vram_free is not None:
                vram_total = vram_used + vram_free
                gpu_pct = gpu_util if gpu_util is not None else min(100.0, 100.0 * vram_used / vram_total)
                used_gb = vram_used / 1024 / 1024 / 1024
                total_gb = vram_total / 1024 / 1024 / 1024
                gpu_note = f"VRAM {used_gb:.1f}/{total_gb:.1f} GiB"
                if gpu_util is not None:
                    gpu_note += f" | GPU {gpu_util}%"
        except Exception:
            pass
        if gpu_pct is None and not self.hw.metal_compute:
            try:
                out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                                     capture_output=True, text=True, timeout=5).stdout
                import re
                m = re.search(r"VRAM \(.*?\):\s*(\d+)\s*(MB|GB)", out, re.IGNORECASE)
                if m:
                    val, unit = int(m.group(1)), m.group(2).upper()
                    total_gb = val / 1024 if unit == "MB" else val
                    gpu_note = f"无 Metal 计算设备 | VRAM {total_gb:.0f} GiB（计算全在 CPU）"
            except Exception:
                pass
        self._draw_bar(cv_c, cpu)
        self._draw_bar(cv_m, mem)
        self._draw_bar(cv_g, gpu_pct, gpu_note)
        pct_c.config(text=f"{cpu:.0f}%")
        pct_m.config(text=f"{mem:.0f}%")
        pct_g.config(text=f"{gpu_pct:.0f}%" if gpu_pct is not None else "n/a")

    # ---------------- 模型库 ----------------
    def _scan_async(self):
        self._log(f"扫描模型库 {self.models_dir} …")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _browse_models_dir(self):
        import tkinter.filedialog as fd
        d = fd.askdirectory(initialdir=self.models_dir, title="选择模型库目录")
        if d:
            self.models_dir = d
            self.dir_var.set(d)
            _save_config(_load_config() | {"models_dir": d})
            self._log(f"模型库已切换 → {d}")
            # 刷新 MTP / mmproj 候选
            global MTP_CANDS, MTP_DISPLAY, MTP_MAP, MMPROJ_CANDS, MMPROJ_DISPLAY, MMPROJ_MAP
            MTP_CANDS, MTP_DISPLAY, MTP_MAP = _build_mtp_cands(d)
            MMPROJ_CANDS, MMPROJ_DISPLAY, MMPROJ_MAP = _build_mmproj_cands(d)
            # 更新 Combobox 候选值
            for key, widget in self.widgets.items():
                if key == "spec_draft_model":
                    widget["values"] = ["default", "off"] + MTP_DISPLAY[1:]
                elif key == "mmproj":
                    widget["values"] = ["default"] + MMPROJ_DISPLAY[1:]
            self._scan_async()

    def _refresh_engbar(self):
        """把配置里的引擎路径回显到 engbar，并保证 Entry 显示「实际生效的路径」。
        规则：配置 vanilla_bin 为空 → 显示自动探测结果（which llama-server → 本地兜底），
        并加提示「(自动)」让主人一眼看出当前用的不是固定路径。"""
        import shutil
        cfg = _load_config()
        v_cfg = cfg.get("vanilla_bin", "").strip()
        t_cfg = cfg.get("tosh_bin", "").strip()
        if v_cfg:
            self.vanilla_var.set(v_cfg)
        else:
            # 空配置 = 自动探测，把探测结果填进输入框让主人可见
            found = shutil.which("llama-server") or os.path.abspath(VANILLA_BIN_FALLBACK)
            self.vanilla_var.set(found + "  (自动探测: 配置为空)")
        if t_cfg:
            self.tosh_var.set(t_cfg)
        else:
            self.tosh_var.set(TOSH_BIN_DEFAULT + "  (默认路径: 配置为空)")
        global VANILLA_ABS, TOSH_ABS
        VANILLA_ABS = v_cfg or os.path.abspath(shutil.which("llama-server") or VANILLA_BIN_FALLBACK)
        TOSH_ABS = t_cfg or TOSH_BIN_DEFAULT

    def _browse_engine(self, which: str):
        import tkinter.filedialog as fd
        cur = self.vanilla_var.get() if which == "vanilla" else self.tosh_var.get()
        p = fd.askopenfilename(initialfile=cur, title=f"选择 {which} 引擎可执行文件")
        if p:
            if which == "vanilla":
                self.vanilla_var.set(p)
            else:
                self.tosh_var.set(p)
            cfg = _load_config()
            cfg[which + "_bin"] = p
            _save_config(cfg)
            global VANILLA_ABS, TOSH_ABS
            if which == "vanilla":
                VANILLA_ABS = p
            else:
                TOSH_ABS = p
            self._refresh_hw_bar()
            self._refresh_engbar()
            self._log(f"引擎路径已更新: {which} → {p}")

    def _apply_engbar(self):
        """engbar「应用」按钮：把两个输入框的文本写进配置并立即生效。
        支持粘贴新路径 / 留空（留空 = 恢复自动探测，升级 llama.cpp 后装进 PATH 即可）。
        输入带「(自动探测)」后缀的自动回显文本时，自动剥离后缀存为配置。"""
        global VANILLA_ABS, TOSH_ABS
        import shutil
        cfg = _load_config()
        v_raw = self.vanilla_var.get().strip()
        t_raw = self.tosh_var.get().strip()
        # 剥离「(自动探测: 配置为空)」后缀（那是 _refresh_engbar 回显的，不是用户输入的）
        v_in = re.sub(r"\s*\(自动探测: 配置为空\)$", "", v_raw).strip()
        t_in = re.sub(r"\s*\(默认路径: 配置为空\)$", "", t_raw).strip()
        if v_in:
            v_in = os.path.abspath(v_in)
            if not os.path.exists(v_in):
                self._log(f"[警告] vanilla 引擎路径不存在: {v_in}")
            cfg["vanilla_bin"] = v_in
            VANILLA_ABS = v_in
        else:
            cfg.pop("vanilla_bin", None)
            VANILLA_ABS = os.path.abspath(shutil.which("llama-server") or VANILLA_BIN_FALLBACK)
        if t_in:
            t_in = os.path.abspath(t_in)
            if not os.path.exists(t_in):
                self._log(f"[警告] tosh 引擎路径不存在: {t_in}")
            cfg["tosh_bin"] = t_in
            TOSH_ABS = t_in
        else:
            cfg.pop("tosh_bin", None)
            TOSH_ABS = TOSH_BIN_DEFAULT
        _save_config(cfg)
        self._refresh_hw_bar()
        self._log("引擎路径已应用" + ("（空 = 自动探测）" if not (v_in or t_in) else ""))

    def _scan_worker(self):
        try:
            models = scan_models(self.models_dir)
        except Exception as e:
            self.q.put(f"[错误] 扫描失败: {e}")
            return
        for st in models:
            self.st_cache[st["path"]] = st
        # 传完整 st 字典（含 layers/active_b），_fill_tree 各列取数不再缺键
        self.q.put("__SCAN_DONE__" + json.dumps(models))

    def _on_select(self):
        sel = self.tree.selection()
        if not sel:
            return
        pid = self.tree.item(sel[0], "values")[-1]  # 最后列存 path
        st = self.st_cache.get(pid)
        if not st:
            return
        self.selected = st
        key = st.get("path")
        saved = (self._load_state().get(key) or {}).get("params") or {}
        if saved:
            # 该模型有用户上次保存的面板值 → 恢复
            self._restore_model_state()
        else:
            # 无历史 → 推荐引擎按本模型算出“尽量装显存/最大 ctx”的预填值写入面板
            self._apply_prefill()
        self._refresh_recommend()

    def _fill_tree(self, models):
        self.tree.delete(*self.tree.get_children())
        self._sort_vals = {}
        for st in models:
            kind = (f"MoE act≈{st.get('active_b',0):.0f}B" if st.get("is_moe") else "dense")
            b = st.get("total_b", 0)
            layers = st.get("layers", 0)
            iid = self.tree.insert("", "end",
                                    values=(st.get("name", "?"),
                                            f"{b:.0f}" if b else "-",
                                            kind,
                                            f"{st.get('file_gb',0):.1f}",
                                            str(layers) if layers else "-",
                                            st.get("path")))
            self._sort_vals[iid] = {"name": st.get("name", "?"),
                                     "b": float(b or 0),
                                     "kind": kind,
                                     "gb": float(st.get("file_gb", 0) or 0),
                                     "layer": int(layers or 0)}
        # 沿用上一次排序状态
        if self._sort_col:
            self._sort_tree()

    def _on_heading_click(self, col):
        """点击表头：同列切换升/降；换列时数值列(B/GB)默认降序（大在前）。"""
        if col == self._sort_col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = col in ("b", "gb")
        self._sort_tree()

    def _sort_tree(self):
        children = self.tree.get_children("")
        if not children:
            return
        col, rev = self._sort_col, self._sort_reverse
        vals = self._sort_vals
        ordered = sorted(children,
                         key=lambda it: vals.get(it, {}).get(col, 0) if not isinstance(
                             vals.get(it, {}).get(col, 0), str)
                         else str(vals.get(it, {}).get(col, 0)).casefold(),
                         reverse=rev)
        for i, child in enumerate(ordered):
            cur = self.tree.index(child)
            if cur != i:
                # tkinter move(item, parent, index) —— 先 item 再 parent，旧版反了
                self.tree.move(child, "", i)
        for c, base in self._heading_base.items():
            if c == col:
                self.tree.heading(c, text=base + (" ▼" if rev else " ▲"))
            else:
                self.tree.heading(c, text=base)

    # ---------------- 推荐 / 命令 ----------------
    def _resolve_engine(self):
        """返回 (engine, binary, notes)。engine_var: auto/vanilla/tos。"""
        if not self.selected:
            return "vanilla", VANILLA_ABS, ("（未选模型）",)
        return recommend.pick_engine(self.selected, self.hw,
                                      forced=self.engine_var.get())

    def _params(self) -> dict:
        p = {k: v.get() for k, v in self.pv.items()}
        # mmproj: 将下拉显示名映射回全路径；若用户手输了全路径则直接透传
        mm = p.get("mmproj", "")
        if mm in MMPROJ_MAP:
            p["mmproj"] = MMPROJ_MAP[mm]
        # 若 mm 不在 MAP 中（用户手输），保持原值（全路径或自定义路径）
        # spec_draft_model: 同 mmproj 逻辑——显示名映射回全路径；default/off→""（FLAG_MAP 守卫省略旗标）
        sm = p.get("spec_draft_model", "")
        if sm in MTP_MAP:
            p["spec_draft_model"] = MTP_MAP[sm]
        p["dynamic_moe"] = "on" if self.dm_var.get() else "off"
        return p

    def _on_ovr_edit(self, _e=None):
        self._refresh_recommend()
        self._save_state_entry()

    # ---------------- 按模型持久化（参数面板/自定义覆盖/引擎） ----------------
    def _state_key(self):
        return self.selected.get("path") if self.selected else None

    def _load_state(self) -> dict:
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state_entry(self):
        """把当前面板/覆盖/引擎按模型路径写回 STATE_FILE；恢复期间与未选模型时跳过。"""
        key = self._state_key()
        if not key or self._restoring:
            return
        # 保存时 mmproj / spec_draft_model 存显示名（用于 Combobox 回显），其余存实际值
        params_to_save = {k: v.get() for k, v in self.pv.items()}
        data = self._load_state()
        data[key] = {"params": params_to_save,
                     "override": self.ovr.get("1.0", "end").strip(),
                     "engine": self.engine_var.get(),
                     "port": self.port_var.get(),
                     "host": self.host_var.get(),
                     "clear_mode": self.clear_mode_var.get() if hasattr(self, "clear_mode_var") else "override"}
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, STATE_FILE)
        except Exception:
            pass   # 持久化失败不阻塞使用

    def _restore_model_state(self):
        """切到某模型时恢复它上次保存的参数；无记录则保持面板当前值。"""
        key = self._state_key()
        if not key:
            return
        entry = self._load_state().get(key) or {}
        saved = entry.get("params") or {}
        # 迁移：旧版单键 cache_type → 拆分后的 K/V 两键（缺省时各按默认）
        if "cache_type" in saved and "cache_type_k" not in saved:
            saved = dict(saved)
            saved.setdefault("cache_type_k", saved["cache_type"])
            saved.setdefault("cache_type_v", saved["cache_type"])
        self._restoring = True
        try:
            for k, v in saved.items():
                if k in self.pv:
                    try:
                        # mmproj: 若存的是全路径，转为显示名
                        if k == "mmproj" and v in MMPROJ_MAP.values():
                            for disp, full in MMPROJ_MAP.items():
                                if full == v:
                                    self.pv[k].set(disp)
                                    break
                        # spec_draft_model: 旧版手存过全路径（或预设/导入带全路径）→ 转显示名回显
                        elif k == "spec_draft_model" and v in MTP_MAP.values():
                            for disp, full in MTP_MAP.items():
                                if full == v:
                                    self.pv[k].set(disp)
                                    break
                        else:
                            self.pv[k].set(str(v))
                    except tk.TclError:
                        pass
            self.ovr.config(state="normal")
            self.ovr.delete("1.0", "end")
            self.ovr.insert("1.0", entry.get("override") or "")
            eng = entry.get("engine")
            if eng in ("auto", "vanilla", "tosh"):
                self.engine_var.set(eng)
            if "dynamic_moe" in saved:
                self.dm_var.set(saved["dynamic_moe"] == "on")
            # 端口/Host/覆盖模式 持久化恢复
            if "port" in entry:
                self.port_var.set(entry["port"])
            if "host" in entry:
                self.host_var.set(entry["host"])
            if "clear_mode" in entry and hasattr(self, "clear_mode_var"):
                self.clear_mode_var.set(entry["clear_mode"])
        finally:
            self._restoring = False

    def _apply_prefill(self):
        """无持久记录时：面板先回到全局默认，再叠加推荐引擎按本模型算出的
        “尽量装显存/最大 ctx”预填值（ngl/ctx/cache_type/fa 等，均可编辑）。
        预填是确定性的，不写回 STATE_FILE（只有用户改动才持久化）。"""
        st = self.selected
        if not st:
            return
        try:
            pre = recommend.prefill(st, self.hw)
        except Exception:
            return
        self._restoring = True
        try:
            # 1) 全部面板项先复位到全局默认，避免残留上一个模型的旧值
            for k in self.pv:
                d = self.param_defaults.get(k)
                if d is None:
                    continue
                try:
                    self.pv[k].set(str(d))
                except tk.TclError:
                    pass
            # 2) 叠加推荐引擎算出的按硬件最优值
            for k, v in pre.items():
                if k in self.pv:
                    try:
                        self.pv[k].set(str(v))
                    except tk.TclError:
                        pass
        finally:
            self._restoring = False

    def _update_ctv_candidates(self):
        """建议#4：ToshLLM 引擎的 cache-type-v 支持 turbo2/3/4（AMD FA 量化），
        按解析后引擎动态增删候选；vanilla 下选过 turbo 值自动回退 default（防 OptionMenu 空白）。"""
        base = ["default", "f16", "q8_0", "q5_1", "q5_0", "q4_0", "iq4_nl"]
        eng = self.engine_var.get()
        if eng == "auto" and self.selected:
            eng = recommend.pick_engine(self.selected, self.hw, forced="auto")[0]
        vals = base + (["turbo2", "turbo3", "turbo4"] if eng == "tosh" else [])
        cur = self.pv["cache_type_v"].get()
        if cur not in vals:
            self.pv["cache_type_v"].set("default")
        w = self.widgets["cache_type_v"]
        if isinstance(w, ttk.Combobox):
            w["values"] = vals
        else:
            # ttk.OptionMenu 无 "values" 选项（直接赋值抛 TclError 中断整条生成链路），
            # 重建下拉菜单项；command 显式写回变量，保证选中即生效。
            menu = w["menu"]  # 下标访问返回 tkinter.Menu 对象（cget 只给路径字符串）
            menu.delete(0, "end")
            var = self.pv["cache_type_v"]
            for v in vals:
                menu.add_command(label=v, command=lambda v=v: var.set(v))

    def _refresh_recommend(self):
        self._update_ctv_candidates()
        st = self.selected
        if not st:
            self.info.config(text="（选择左侧模型以查看推荐参数）", fg="#888")
            self._cmd_available = False
            self._refresh_btns()
            return
        hw = self.hw
        p = self._params()
        try:
            ctx = int(p["ctx"])
        except (ValueError, tk.TclError):
            ctx = 65536
        ncmoe = None
        # 11-5 规范：n-cpu-moe 数字参数已删除；cpu_moe=on（专家全上 CPU）时按满层数估算
        if str(p.get("cpu_moe", "off")).strip() == "on" and st.get("layers"):
            ncmoe = int(st["layers"])
        # K/V 拆分后取 K 侧为代表性量化（缺省时回退旧键 cache_type）
        kv = p.get("cache_type_k") or p.get("cache_type") or "q8_0"
        plan = recommend.recommend(st, hw, ctx=ctx,
                                   kv_type=kv, ncmoe=ncmoe)

        engine, binary, eng_notes = self._resolve_engine()
        eng_lines = list(eng_notes) if isinstance(eng_notes, (list, tuple)) else eng_notes.splitlines()
        if not binary or not os.path.exists(binary):
            self.info.config(
                text=f"推荐引擎 {engine} 的二进制缺失：{binary or '(无)'}\n" + "\n".join(eng_lines),
                fg="#c0392b")
            self._cmd_available = False
            self._refresh_btns()
            return

        extra = [x for x in (st.get("extra_flags") or "") if x]
        override = self.ovr.get("1.0", "end").strip()
        try:
            port = int(self.port_var.get() or 8080)
        except (ValueError, tk.TclError):
            port = 8080
        host = self.host_var.get().strip() or "127.0.0.1"
        clear_mode = self.clear_mode_var.get() if hasattr(self, "clear_mode_var") else "override"
        cmd = recommend.build_command(st, hw, plan, binary,
                                      port=port, host=host, extra=extra, params=p,
                                      override_args=override,
                                      clear_mode=clear_mode)
        # 计算被覆盖的冲突族用于提示
        import shlex
        user_tokens = shlex.split(override or "")
        fams = set()
        for tok in user_tokens:
            if tok.startswith("-"):
                fam = recommend._flag_family(tok)
                if fam:
                    fams.add(fam)
        hit_names = sorted("或".join(sorted(f)) for f in fams)

        cmd_str = " ".join(cmd)
        self._last_cmd = cmd_str
        self.cmd.config(state="normal")
        self.cmd.delete("1.0", "end")
        self.cmd.insert("1.0", cmd_str)
        self.cmd.config(state="disabled")

        if hit_names:
            self.ovr_hint.config(
                text="自定义参数已覆盖面板同名参数：" + "、".join(hit_names),
                foreground="#c0392b")
        elif override:
            self.ovr_hint.config(text="自定义参数与面板无冲突，按原样追加",
                                 foreground="#666")
        else:
            self.ovr_hint.config(text=" ", foreground="#666")

        lines = [f"引擎: {engine}（binary: {os.path.basename(binary)}）"]
        lines += [f"  - {n}" for n in eng_lines if n]
        lines.append(
            f"模型 ≈ {st.get('file_gb', 0):.1f} GiB（{plan.get('kv_type','q8_0')} KV，ctx={ctx}）")
        # 硬件描述不再重复（顶部硬件栏已显示）
        if plan.get("plan_notes"):
            lines.append("计划: " + plan["plan_notes"])
        if plan.get("swa_note"):
            lines.append(plan["swa_note"])
        if plan.get("dynamic_moe_note"):
            lines.append(plan["dynamic_moe_note"])
        cpu_moe_on = str(p.get("cpu_moe", "off")).strip() == "on"
        hints = [h for h in (plan.get("hints") or []) if h]
        if cpu_moe_on:  # --cpu-moe 全专家已上 CPU，OOM 提示不再适用
            hints = [h for h in hints if not h.startswith("⚠️ OOM")]
        for h in hints:
            lines.append("提示: " + h)
        # 与提示内容重复的告警不再重打（去重）
        warns = [w for w in (plan.get("warnings") or [])
                 if not any(h and h in w for h in hints)]
        if warns:
            lines.append("⚠ " + "; ".join(warns))
        if st.get("mmap_ok") is False:
            lines.append("警告: 该模型无法 mmap（NVMe/网络卷）→ 将整读入 RAM")
        if engine == "tosh" and str(p.get("moe_cache_mib")) not in ("0", "auto", "default", "不配置", "") and str(p.get("moe_cache_cpu_share")) not in ("0.0", "auto", "default", "不配置", ""):
            lines.append("注: tosh 引擎无 moe-cache-* 旗标，该两项不会出现在命令里")
        if self.dm_var.get() and st.get("is_moe"):
            n = "本机 dynamic-moe 规划器易因 VRAM/coverage 拒载，拒载后自动以 off 重试"
            if str(p.get("cpu_moe", "off")).strip() == "on":
                n += "；且 --cpu-moe（专家全上 CPU）与 --dynamic-moe 语义冲突，建议 cpu-moe 选 off"
            lines.append("注: " + n)
        # 过滤：不显示 "⚠"/"警告"/"清空模式" 开头的行（已进主日志）
        filtered = [ln for ln in lines if not ln.startswith(("⚠", "警告", "清空模式"))]
        oom_on = bool(plan.get("oom_risk")) and not cpu_moe_on
        self.info.config(text="\n".join(filtered) or "（选择左侧模型以查看推荐参数）",
                         fg="#c62828" if oom_on else "#c77700")
        # 命令已生成：点亮 ⚡生成指令 / 复制 两钮
        self._cmd_available = bool(cmd_str.strip())
        self._refresh_btns()

    def _gen_cmd(self):
        """手动触发命令生成（刷新命令框）。若未选模型自动选第一个。"""
        if not self.selected:
            # 自动补选：取 tree 第一个可见项
            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
                self._on_select()
            if not self.selected:
                # tree 空 = 扫描尚未完成，直接等扫描线程写完 queue
                self._log("模型库尚未扫描完成，请稍候再点「⚡ 生成指令」。")
                return
        self._refresh_recommend()
        self._log("已刷新命令")

    def _on_copy_click(self):
        """复制钮入口：未生成命令时提示；已生成则复制到剪贴板。"""
        if not self._cmd_available:
            messagebox.showinfo("未生成", "请先生成命令（选择模型后点「⚡ 生成指令」）。")
            return
        self._do_copy()

    def _do_copy(self):
        txt = self.cmd.get("1.0", "end").strip()
        if not txt:
            self._log("命令框为空，未复制")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self._log("已复制命令到剪贴板")
        self._refresh_btns()

    # ---------------- 配置另存（预设） / 导出 / 导入 ----------------
    def _current_state(self) -> dict:
        """捕获当前完整配置：参数面板 + 自定义覆盖 + 引擎 + 端口 + host + dynamic-moe。"""
        return {
            "params": {k: v.get() for k, v in self.pv.items()},
            "override": self.ovr.get("1.0", "end").strip(),
            "engine": self.engine_var.get(),
            "port": self.port_var.get(),
            "host": self.host_var.get(),
            "dynamic_moe": bool(self.dm_var.get()),
        }

    def _apply_state(self, state: dict):
        """把一份配置状态套到当前面板（预设/导入共用），再刷新命令。"""
        state = state or {}
        self._restoring = True
        try:
            params = state.get("params") or {}
            for k, var in self.pv.items():
                if k in params:
                    try:
                        var.set(str(params[k]))
                    except tk.TclError:
                        pass
            self.ovr.config(state="normal")
            self.ovr.delete("1.0", "end")
            self.ovr.insert("1.0", state.get("override") or "")
            eng = state.get("engine")
            if eng in ("auto", "vanilla", "tosh"):
                self.engine_var.set(eng)
            if "dynamic_moe" in state:
                self.dm_var.set(bool(state.get("dynamic_moe")))
            if state.get("port"):
                self.port_var.set(str(state["port"]))
            if state.get("host"):
                self.host_var.set(str(state["host"]))
        finally:
            self._restoring = False
        self._refresh_recommend()

    def _load_presets(self) -> dict:
        try:
            with open(PRESET_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_presets(self, presets: dict):
        tmp = PRESET_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=1)
        os.replace(tmp, PRESET_FILE)

    def _refresh_preset_names(self):
        names = sorted(self._load_presets().keys())
        cur = self.preset_var.get()
        self.preset_combo["values"] = names
        if cur in names:
            self.preset_combo.set(cur)
        elif names:
            self.preset_combo.set(names[0])
        self.preset_hint.config(text=f"{len(names)} 个预设" if names else "（无预设）")

    def _preset_load(self):
        name = self.preset_var.get().strip()
        entry = self._load_presets().get(name)
        if not entry:
            messagebox.showwarning("加载预设", "无此预设或未选择。")
            return
        self._apply_state(entry)
        self._log(f"已加载预设「{name}」")
        self._save_state_entry()   # 若已选模型，顺带把套上的配置按模型落盘

    def _preset_save_as(self):
        init = ((self.selected or {}).get("name") or "我的预设").replace(".gguf", "")
        name = simpledialog.askstring("另存预设", "预设名称（可跨模型复用）：",
                                     parent=self.root, initialvalue=init)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        presets = self._load_presets()
        presets[name] = self._current_state()
        try:
            self._save_presets(presets)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return
        self._refresh_preset_names()
        self.preset_combo.set(name)
        self._log(f"已另存预设「{name}」→ {os.path.basename(PRESET_FILE)}")

    def _preset_delete(self):
        name = self.preset_var.get().strip()
        presets = self._load_presets()
        if name not in presets:
            self._log(f"（无此预设，未删除）「{name}」" if name else "未选择预设，未删除")
            return
        del presets[name]
        self._save_presets(presets)
        self._refresh_preset_names()
        self._log(f"已删除预设「{name}」")

    def _export(self):
        model_name = ((self.selected or {}).get("name") or "llama-shell").replace(".gguf", "")
        default = f"{model_name}_llama-shell.json"
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".json", initialfile=default,
            filetypes=[("JSON", "*.json")], title="导出配置")
        if not path:
            return
        payload = {
            "app": "llama-shell",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_model": (self.selected or {}).get("path", ""),
            "state": self._current_state(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
            return
        self._log(f"已导出配置 → {os.path.basename(path)}")

    def _import(self):
        path = filedialog.askopenfilename(
            parent=self.root, filetypes=[("JSON", "*.json"), ("All", "*")],
            title="导入配置")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("导入失败", f"无法解析 {os.path.basename(path)}：{e}")
            return
        # 兼容：导出封装（含 state）或裸状态 dict
        state = data.get("state") if isinstance(data, dict) and "state" in data else data
        if not isinstance(state, dict) or "params" not in state:
            messagebox.showerror("导入失败", "文件格式不符：缺少 params 字段。")
            return
        self._apply_state(state)
        self._log(f"已导入配置 ← {os.path.basename(path)}")
        self._save_state_entry()

    # ---------------- 运行 / 停止 / 测速 ----------------
    def _on_run_click(self):
        """运行钮点击：无存活进程才允许启动。"""
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("运行中", "已有进程在运行，请先停止。")
            return
        self._run()

    def _on_stop_click(self):
        """停止钮点击：有存活进程才允许停止。"""
        if self.proc and self.proc.poll() is None:
            self._stop()
        else:
            self._log("当前无运行中的进程，无需停止。")

    def _on_gen_click(self):
        """⚡生成指令钮点击：刷新命令（未选模型自动补选）。"""
        self._gen_cmd()

    def _run(self):
        # 点击层已拦截：仅在无存活进程时才会调用到这里
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("运行中", "已有进程在运行，请先停止。")
            return
        import shlex
        cmd_str = self.cmd.get("1.0", "end").strip()
        if not cmd_str:
            messagebox.showwarning("无命令", "请先生成命令（选择模型）。")
            return
        self._auto_retried = False
        self._spawn(shlex.split(cmd_str))

    def _spawn(self, cmd):
        # 兼容：cmd 可以是 list（推荐路径）或字符串（__RETRY__ 旧路径）
        if isinstance(cmd, str):
            import shlex
            cmd_list = shlex.split(cmd)
        else:
            cmd_list = list(cmd)
        cmd_str = " ".join(cmd_list)
        self._last_cmd = cmd_str
        self._dynamic_moe_used = "--dynamic-moe on" in cmd_str
        self._log(f"启动: {cmd_str}")
        # 11-13 自愈：bind 前清掉占用端口的残留进程（上次会话孤儿/停不掉的 35B），
        # 否则新 server 加载即失败。kill_port 只会命中 lsof 上该端口的监听者。
        try:
            port = int(self.port_var.get() or 8080)
        except Exception:
            port = 8080
        self._last_port = port
        pids = kill_port(port)
        if pids:
            self._log(f"已先清理占用端口 {port} 的残留进程 PIDs {pids}")
        try:
            # shell=False + start_new_session=True：self.proc 直接指向 llama-server，
            # os.killpg 能精准杀掉它（含其工作线程）。旧 shell=True 会让 self.proc
            # 指向 /bin/sh，killpg 只杀到 shell，llama-server 成孤儿继续占端口/内存。
            self.proc = subprocess.Popen(
                cmd_list, shell=False, cwd=BASE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                text=True, bufsize=1)
        except Exception as e:
            self._log(f"[错误] 启动失败: {e}")
            return
        # 关键：_ready 必须等 HEALTH ok 才置位（_on_health），这里只能回灰，
        # 绝不能直接置 ready——否则点击瞬间按钮就会变绿（"一按就绿"的根因）
        self._ready = False
        self._reader_stop = False
        self._health_stop = False
        self._refresh_btns()          # 画灰（● 正在加载）；绿色由 _on_health 点亮
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._health_poll, daemon=True).start()
        self._set_run_ui(True)
        self.status.config(text="运行中…", foreground="#c77700")

    def _reader(self):
        p = self.proc
        for line in iter(p.stdout.readline, ""):
            self.q.put("OUT:" + line)
            # 自动回退：tosh 规划器拒载（VRAM/coverage）→ 用 dynamic-moe off 重跑一次
            if ("TOSH_AUTO found no configuration that fits" in line
                    or "PLAN_UNSUPPORTED" in line):
                if self._dynamic_moe_used and not self._auto_retried:
                    self._auto_retried = True
                    self.q.put("OUT:\n[GUI] Dynamic MoE 被规划器拒载（VRAM/coverage 不足）— 自动以 --dynamic-moe off 重试 …\n")
                    retry = self._last_cmd.replace("--dynamic-moe on", "--dynamic-moe off")
                    self.q.put("__RETRY__:" + retry)
        rc = p.wait()
        self.q.put(f"PROC_EXIT:{rc}")

    def _health_poll(self):
        try:
            port = int(self.port_var.get() or 8080)
        except (ValueError, tk.TclError):
            port = 8080
        host = self.host_var.get().strip() or "127.0.0.1"
        last_ok = None
        while not self._health_stop:
            p = self.proc      # 实时读取：dynamic-moe 自动重试换新进程后，盯住新进程继续轮询
            if p is None or p.poll() is not None:
                break          # 无进程 / 进程已退（_stop/PROC_EXIT 会把 proc 置 None）
            ok = health(port, host)
            # 仅在结果变化时入队 → 事件不再每 1s 灌满队列；_poll 循环保持轻载
            if ok is not last_ok:
                self.q.put("HEALTH:" + ("1" if ok else "0"))
                last_ok = ok
            time.sleep(1.0)

    def _stop(self):
        self._reader_stop = True
        self._health_stop = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            self._log("已发送 SIGTERM（进程组），后台等待退出…")
            # 11-13：等待/SIGKILL/清端口全部移入后台 _reap 线程——
            # 35B 大模型 SIGTERM 慢时，主线程不再卡 5s，停止按钮/关窗立即可用。
            threading.Thread(target=self._reap, args=(proc,), daemon=True).start()
        self.proc = None  # 立即清空，运行钮瞬间可用；旧进程退出后 PROC_EXIT 会再次置 None 无害
        self._ready = False  # 已停止 → 运行钮恢复灰色（须健康轮询再次判定 ready）
        self._refresh_btns()
        # 清理端口：杀残留占用进程（11-13：优先 _last_port——atexit 场景下 tk 可能已销毁）
        try:
            port = int(self.port_var.get() or 8080)
        except (ValueError, tk.TclError):
            port = getattr(self, "_last_port", 8080)
        self._last_port = port
        killed = kill_port(port)
        if killed:
            self._log(f"已清理端口 {port}（PID: {killed}）")
        self._set_run_ui(False)
        self.status.config(text="停止中…", foreground="#c77700")

    def _reap(self, proc):
        """后台回收线程（11-13）：SIGTERM 后最多等 5s → 仍存活则 SIGKILL 进程组 → 清端口。
        全程不碰任何 widget，只经 q.put 回传日志；主线程 destroy 后消息静默丢弃，atexit 兜底接力。"""
        for _ in range(25):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self.q.put("OUT:\n[GUI] 进程未退出，已发送 SIGKILL（进程组）强制结束\n")
            time.sleep(0.3)
        try:
            killed = kill_port(getattr(self, "_last_port", 8080))
            if killed:
                self.q.put(f"OUT:\n[GUI] 已清理端口 {self._last_port} 残留（PID: {killed}）\n")
        except Exception:
            pass
        self.q.put("OUT:\n[GUI] 已停止\n")

    def _on_close(self):
        """窗口关闭：非阻塞地停掉 server，立即销毁窗口。

        若 server 正在运行：发 SIGTERM 到进程组，后台清理，不等它退出。
        这样 macOS 用户点关闭按钮后窗口立即消失，不会卡 5 秒。
        """
        self._closing = True
        self._reader_stop = True
        self._health_stop = True
        # 取消所有 pending after 定时器，防止 destroy 后回调再入主循环
        for attr in ("_poll_id", "_hw_tick_id"):
            try:
                tid = getattr(self, attr, None)
                if tid:
                    self.root.after_cancel(tid)
            except Exception:
                pass
        # 若有 server 在跑：发 SIGTERM（不阻塞等待），靠后台 reader 线程收尾
        if self.proc is not None and self.proc.poll() is None:
            try:
                import signal, os
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self._log("正在停止 server…（后台）")
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        # 兜底清理端口：杀残留的 llama-server 孤儿（dynamic-moe 切换后 shell 死了但 server 还在）
        try:
            port = int(self.port_var.get() or 8080)
        except (ValueError, tk.TclError):
            port = getattr(self, "_last_port", 8080)
        self._last_port = port
        try:
            killed = kill_port(port)
            if killed:
                self._log(f"已清理端口 {port} 残留（PID: {killed}）")
        except Exception:
            pass
        try:
            self._save_state_entry()
        except Exception:
            pass
        self.root.destroy()

    def _bench(self):
        try:
            port = int(self.port_var.get() or 8080)
        except (ValueError, tk.TclError):
            port = 8080
        host = self.host_var.get().strip() or "127.0.0.1"
        if not health(port, host):
            self._log(f"[测速] {host}:{port} 无响应，先运行 server。")
            return
        self.status.config(text="测速中…", foreground="#c77700")
        threading.Thread(target=self._bench_worker, args=(port, host), daemon=True).start()

    def _bench_worker(self, port, host):
        try:
            tps = bench_once(port, host)
            self.q.put(f"OUT:\n[测速] 生成 48 token → {tps:.1f} t/s\n")
        except Exception as e:
            self.q.put(f"OUT:\n[测速] 失败: {e}\n")

    # ---------------- 轮询（主线程） ----------------
    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        if getattr(self, "_closing", False):
            return
        try:
            while True:
                item = self.q.get_nowait()
                if item == "__NOOP__":
                    break
                if item.startswith("__SCAN_DONE__"):
                    self._fill_tree(json.loads(item[len("__SCAN_DONE__"):]))
                    self._log(f"已载入 {len(self.st_cache)} 个模型")
                elif item.startswith("OUT:"):
                    self._log(item[4:])
                elif item.startswith("HEALTH:"):
                    ok = item[7:] == "1"
                    # 状态机：HEALTH 健康 = 系统 ready（运行钮可点亮的门）
                    self._ready = ok
                    if hasattr(self, "status"):
                        self.status.config(
                            text="READY" if ok else "运行中…",
                            foreground="#1a7f37" if ok else "#c77700")
                    self._refresh_btns()
                elif item.startswith("PROC_EXIT:"):
                    # 兼容旧切片写法：item[10:] 是字符串，不是索引
                    self.proc = None
                    self._ready = False        # 进程退出 → 不再 ready，运行钮恢复灰色
                    self.status.config(text="已退出", foreground="#666")
                    self._set_run_ui(False)
                    self._refresh_btns()
                elif item.startswith("__RETRY__:"):
                    retry_cmd = item[len("__RETRY__:"):].strip()
                    def do_retry():
                        if self._closing:
                            return
                        if self.proc is None:
                            return          # 用户已停止/关闭，放弃重试
                        if self.proc.poll() is None:
                            self.root.after(300, do_retry)   # 等旧进程退出再换
                            return
                        self.cmd.config(state="normal")
                        self.cmd.delete("1.0", "end")
                        self.cmd.insert("1.0", retry_cmd)
                        self.cmd.config(state="disabled")
                        self._spawn(retry_cmd)
                    do_retry()
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self._poll_id = self.root.after(80, self._poll)


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 2.0)  # 高分屏放大
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()