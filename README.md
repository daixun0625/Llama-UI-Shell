# Llama UI Shell

> 轻量级 Tkinter GUI，封装 `llama.cpp` (b12212) 与 **ToshLLM**，专为本地大模型推理设计 —— 模型浏览、硬件感知自动参数推荐、双引擎一键启停、按模型持久化配置。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **双引擎原生支持** | Vanilla `llama.cpp` (b12212) + **ToshLLM** (Apple Silicon / AMD eGPU 优化) |
| **硬件感知自动推荐** | 一键读取 VRAM/RAM/CPU/Metal，按模型架构 (MoE / 稠密) 计算最优 `ngl` / `n-cpu-moe` / `moe-cache-mib` / `ctx` / `kv-type` 等 |
| **MoE 全参数覆盖** | `--cpu-moe` / `--n-cpu-moe` / `--n-cpu-ffn` / `--moe-cache-mib` / `--moe-cache-cpu-share` / `--dynamic-moe` (Tosh 专用) |
| **推测解码 / MTP** | `--spec-type draft-mtp` / `--spec-draft-model` / `--spec-draft-n-max` / `--cache-type-k-draft` / `--cache-type-v-draft` (双引擎均支持)；MTP 独立 draft 文件自动识别（文件名含 mtp，大小写不敏感，如 `mtpdraft-*`），不进主模型库 |
| **KV 量化分离** | `--cache-type-k` / `--cache-type-v` 独立控制 (默认 q8_0，约 53% f16 显存) |
| **多模态投影 (mmproj)** | 自动扫描 `mmproj*.gguf`，下拉选择，`--mmproj-device` 可选 cpu/gpu/none |
| **按模型持久化** | `gui_state.json` 以模型绝对路径为 key，保存面板全量参数、自定义 override、引擎选择、端口/绑定地址 |
| **预设 / 导出 / 导入** | `gui_presets.json` 跨模型复用；JSON 文件导出导入，便于备份/迁移/分享 |
| **端口自动清理** | 关模型 / 关窗口自动 `kill_port()`，彻底释放占用 |
| **macOS 安全关窗** | `WM_DELETE_WINDOW` 协议处理，避免 Tk 8.6 死锁 |
| **实时硬件监控** | psutil (CPU/RAM) + ioreg (AMD GPU VRAM/利用率) 2 秒刷新 |
| **零第三方依赖** | 纯 `tkinter` + 标准库，`python3 gui.py` 即跑 |

---

## 🖥 硬件基线 (开发/测试环境)

| 组件 | 型号 |
|------|------|
| CPU | Intel i5-12400F (6C/12T) |
| 内存 | 32 GB DDR4 |
| GPU | AMD Radeon RX 6600 XT **8 GB VRAM** (macOS Metal / Vulkan) |
| 系统 | macOS 15 (Intel x64) / Windows 11 (双机同配置) |

> **典型跑分** (35B MoE / Q4_K_M / ctx=65536 / kv=q8_0 / fa=on)  
> - ToshLLM: **~24 tok/s**  
> - Vanilla b12212: **~14 tok/s**  
> 加速主要来自 ToshLLM 的 CPU MoE 内核而非 `--dynamic-moe`。

---

## 🚀 快速开始

```bash
# 1. 克隆 / 下载
git clone <your-repo-url> llama-shell
cd llama-shell

# 2. 准备二进制 (二选一)
#   A) Vanilla b12212
#      解压 llama-b12212-macos-x64.tar.gz → bin/llama-b12212/llama-server
#   B) ToshLLM
#      安装 ToshLLM.app → /Applications/ToshLLM.app/Contents/Resources/bin/llama-server

# 3. 配置路径 (首次运行会弹窗引导，或手动编辑)
cat > llama-shell-config.json <<'EOF'
{
  "models_dir": "/Volumes/CT-nvme/llama-bin-win-vulkan-x64/models",
  "vanilla_bin": "/Users/dx/llama-shell/bin/llama-b12212/llama-server",
  "tosh_bin": "/Applications/ToshLLM.app/Contents/Resources/bin/llama-server"
}
EOF

# 4. 运行
python3 gui.py
```

> **macOS 桌面快捷方式**  
> `~/Desktop/llama-shell.app` (原生 `.app` bundle，`/usr/local/bin/python3` 硬编码，Finder 双击即启)

---

## 📋 界面布局

```
┌─────────────────────────────────────────────────────────────────────┐
│ 工具栏: 引擎选择 | 端口 | 绑定地址 | ▶运行 | ■停止 | 版权声明           │
├──────────────────┬──────────────────────────────────────────────────┤
│                  │ ① CPU / MoE 参数                                 │
│   模型库 (10行)  │     ngl / threads / n-cpu-moe / moe-cache-mib   │
│                  │     n-cpu-ffn / moe-cache-cpu-share / dynamic-moe│
│ ──────────────── │ ② 缓存 / 解码 参数                               │
│ 内存 / 速度 / 告警 │     ctx / kv-type(k/v分离) / flash-attn        │
│   (白字、同底色)  │     batch / ubatch / mmproj / mmproj-device     │
│ ──────────────── │     spec-type / spec-draft-model / spec-draft-n │
│     日志 (32行)  │     spec-cache-type-k/v                         │
│                  │ ③ 生成 / 采样 参数                               │
│                  │     temp / top_p / top_k / min_p                │
│                  │     repeat-penalty / presence-penalty / seed    │
│                  │     思考模式开关 (三态: auto/off/on)             │
├──────────────────┴──────────────────────────────────────────────────┤
│ 硬件负载 (实时 2s)  |  后端命令框 (15行, 原生灰滚动条)                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 关键参数速查

| 参数 | 位置 | 说明 | 推荐值 (8GB VRAM + 32GB RAM) |
|------|------|------|------------------------------|
| `ngl` | ① | GPU 卸载层数，`auto`=由推荐器决定 | MoE: `999` (全卸载)+`n-cpu-moe`；Dense: `auto` |
| `n-cpu-moe` | ① | MoE 专家留 CPU 层数 | `32` (35B) / `auto` |
| `moe-cache-mib` | ① | MoE 专家缓存 (MiB，仅 Vanilla) | `4096`~`8192` |
| `moe-cache-cpu-share` | ① | 专家缓存 CPU 占比 (仅 Vanilla) | `0.5` |
| `dynamic-moe` | ① | ToshLLM 专用动态 MoE | `on` (Tosh) |
| `ctx` | ② | 上下文长度 | `65536` (首选) / `32768` / `16384` |
| `cache-type-k/v` | ② | KV 量化 (分离) | `q8_0` (默认) / `q4_0` / `f16` |
| `flash-attn` | ② | Flash Attention | `on` (MoE 大 ctx 必开) |
| `batch/ubatch` | ② | 并行/微批大小 | MoE: `4096/1024`；Dense: `2048/512` |
| `mmproj` | ② | 多模态投影文件 | 下拉选择 `mmproj*.gguf` |
| `spec-type` | ② | 推测解码类型 | `draft-mtp` |
| `temp / top_p` | ③ | 采样温度 / 核采样 | `0.75 / 0.95` |
| `repeat-penalty` | ③ | 重复惩罚 | `1.1` |
| `presence-penalty` | ③ | 存在惩罚 | `0.05` |

---

## 🔧 推荐器算法 (`recommend.py`)

```python
# 入口
plan = recommend.recommend(st, hw, ctx=65536, kv_type="q8_0", ncmoe="auto")

# 核心逻辑 (prefill)
# 1. 尝试 ctx 候选 [65536, 32768, 16384, 8192, 4096] 降序
# 2. MoE: -ngl 999 + --n-cpu-moe N (把专家塞进 VRAM)
# 3. Dense: 全 GPU 能装则全装，否则半量；纯 CPU 按 RAM 预算 (72% - 2GB)
# 4. 选最大装得下的 ctx，返回完整面板字典供 GUI 直接套用
```

**输出字段**: `ngl`, `ncpu_moe`, `moe_cache_mib`, `moe_cache_cpu_share`, `ncpu_ffn`, `cpu_moe`, `ctx_size`, `kv_type`, `threads`, `batch_size`, `ubatch_size`, `offload_kqv`, `flash_attn`, `warnings`, `level`, `hints`, `engine_specific`

---

## 📦 配置文件结构

### `llama-shell-config.json` (全局路径)
```json
{
  "models_dir": "/path/to/models",
  "vanilla_bin": "/path/to/llama-b12212/llama-server",
  "tosh_bin": "/path/to/ToshLLM/llama-server"
}
```

### `gui_state.json` (按模型持久化)
```json
{
  "/abs/path/to/model.gguf": {
    "params": { "ngl": "999", "ncmoe": "32", "ctx": "65536", ... },
    "override": "-ngl 999 --n-cpu-moe 32 --cache-type-k q8_0 ...",
    "engine": "tosh",
    "port": 8080,
    "host": "0.0.0.0"
  }
}
```

### `gui_presets.json` (跨模型预设)
```json
{
  "my-moe-preset": {
    "params": { ... },
    "override": "",
    "engine": "tosh",
    "port": 8080,
    "host": "127.0.0.1",
    "dynamic_moe": true
  }
}
```

---

## 🧪 测试套件

```bash
# 状态持久化 E2E (5 场景)
python3 test_state.py

# 推荐器 prefill 管线 (MoE + Dense)
python3 test_prefill.py

# 预设/导出/导入 (7 场景)
python3 test_preset.py
```

> 全绿通过 ✅

---

## 📁 目录结构

```
llama-shell/
├── gui.py                  # 主程序 (纯 Tkinter, ~1200 行)
├── recommend.py            # 推荐器核心 (~35 KB)
├── gguf_fast.py            # GGUF 极速解析 (混合层/KV/SSM 估算)
├── gguf_parse.py           # GGUF 基础解析
├── llama-shell-config.json # 全局路径配置
├── gui_state.json          # 按模型状态 (运行时生成)
├── gui_presets.json        # 用户预设 (运行时生成)
├── test_state.py
├── test_prefill.py
├── test_preset.py
├── bench.py                # 简易基准
├── bin/
│   └── llama-b12212/       # Vanilla 二进制 (自备)
└── progress/
    └── llama-shell.md      # 开发台账 (中文)
```

---

## 🛠 进阶用法

### 自定义启动参数 (Override)
右下「自定义参数」文本框，逐行写入额外 CLI 标志，例如：
```
--mlock
--no-mmap
--cont-batching
--swa-full
```
会追加在自动生成命令之后。

### 局域网暴露
工具栏「绑定地址」改为 `0.0.0.0`，端口默认 `8080`，防火墙放行即可远程调用 OpenAI 兼容 API。

### 多模型切换
左侧列表双击 → 自动跑推荐器 → 面板刷新 → 点「运行」。  
切换时自动 `kill_port()` 清理旧端口，再拉新进程。

---

## 🐛 已知限制 / TODO

- [ ] **按钮渲染** macOS Aqua 主题下 Canvas 绘制的运行/停止按钮文字极淡 (需更深层主题覆盖)
- [ ] **思考模式按钮** 统一为 "Thinking Mode" 风格三态开关 (目前 OptionMenu 实现)
- [ ] **日志再增 10 行** (现 32 行)
- [ ] **窗口整体上移 7 行** (现 954px → 目标 ~900px)
- [ ] **MTP 推测解码** 实测效果验证 (双引擎均已支持参数)
- [ ] **Windows 原生打包** (pyinstaller / msix)
- [ ] **Linux / Wayland 适配** (Tk 8.6 表现待测)

---

## 📄 许可证

MIT License — 代码完全开源，二进制 `llama.cpp` / `ToshLLM` 遵循各自上游协议 (MIT / Apache-2.0 / LGPL 等)。

---

## 🙏 致谢

- **llama.cpp** (ggerganov) — 本地推理基石
- **ToshLLM** — Apple Silicon / AMD eGPU 深度优化
- **社区经验** — AMD RX 6600 XT 8GB VRAM + 32GB RAM 实测参数总结

---

> **一句话定位**：给「想跑大模型、不想折腾参数、要双引擎对比、要配置落盘」的开发者/研究者用的 **本地轻量控制台**。