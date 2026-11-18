# llama-shell v0.1.0 — 首个公开 Release

> **轻量级 Tkinter GUI，封装 llama.cpp (b12212) 与 ToshLLM**，专为本地大模型推理设计 —— 模型浏览、硬件感知自动参数推荐、双引擎一键启停、按模型持久化配置。

---

## 📦 下载

| 平台 | 文件 | 说明 |
|------|------|------|
| macOS (Intel x64) | `llama-shell-macos-x64.tar.gz` | 含 `bin/llama-b12212/llama-server` (Vanilla)；需自行安装 ToshLLM.app |
| 源码 | `Source code (zip/tar.gz)` | 纯 Python，零第三方依赖 |

> **Windows / Linux** 暂未提供预构建包，直接 `python3 gui.py` 即可运行（需自备对应平台二进制）。

---

## ✨ v0.1.0 亮点

### 核心功能
- ✅ **双引擎原生支持** — Vanilla `llama.cpp` b12212 + **ToshLLM** (AMD eGPU / Apple Silicon 深度优化)
- ✅ **硬件感知自动推荐** — 一键读取 VRAM/RAM/CPU/Metal，按模型架构 (MoE / 稠密) 计算最优参数
- ✅ **MoE 全参数覆盖** — `n-cpu-moe` / `moe-cache-mib` / `moe-cache-cpu-share` / `dynamic-moe` (Tosh 专用) / `n-cpu-ffn` / `cpu-moe`
- ✅ **推测解码 / MTP** — `spec-type draft-mtp` / `spec-draft-model` / `spec-draft-n-max` / `spec-cache-type-k/v` (双引擎均支持)；MTP 独立 draft 文件自动识别（文件名含 mtp，大小写不敏感，如 `mtpdraft-*`），不进主模型库
- ✅ **KV 量化分离** — `cache-type-k` / `cache-type-v` 独立控制 (默认 q8_0，约 53% f16 显存)
- ✅ **多模态投影 (mmproj)** — 自动扫描 `mmproj*.gguf`，下拉选择，`mmproj-device` 可选 cpu/gpu/none
- ✅ **按模型持久化** — `gui_state.json` 以模型绝对路径为 key，保存面板全量参数、自定义 override、引擎选择、端口/绑定地址
- ✅ **预设 / 导出 / 导入** — `gui_presets.json` 跨模型复用；JSON 文件导出导入，便于备份/迁移/分享
- ✅ **端口自动清理** — 关模型 / 关窗口自动 `kill_port()`，彻底释放占用
- ✅ **macOS 安全关窗** — `WM_DELETE_WINDOW` 协议处理，避免 Tk 8.6 死锁
- ✅ **实时硬件监控** — psutil (CPU/RAM) + ioreg (AMD GPU VRAM/利用率) 2 秒刷新
- ✅ **零第三方依赖** — 纯 `tkinter` + 标准库，`python3 gui.py` 即跑

### 典型硬件跑分 (开发环境)
| 硬件 | 模型 | 引擎 | 速度 |
|------|------|------|------|
| i5-12400F + RX 6600 XT 8GB + 32GB RAM | Huihui-Qwen3.6-35B-A3B (Q4_K_M) | ToshLLM | **~24 tok/s** |
| 同上 | 同上 | Vanilla b12212 | **~14 tok/s** |
| 同上 | Arete-Qwen2.5-27B (Q4_K_M) | ToshLLM | **~18 tok/s** |
| 同上 | 同上 | Vanilla b12212 | **~12 tok/s** |

> 加速主要来自 ToshLLM 的 CPU MoE 内核而非 `--dynamic-moe`。

---

## 🚀 快速上手

```bash
# 1. 解压 / 克隆
tar -xzf llama-shell-macos-x64.tar.gz
cd llama-shell

# 2. 配置路径 (首次运行会弹窗引导)
cat > llama-shell-config.json <<'EOF'
{
  "models_dir": "/Volumes/你的盘/models",
  "vanilla_bin": "/path/to/llama-shell/bin/llama-b12212/llama-server",
  "tosh_bin": "/Applications/ToshLLM.app/Contents/Resources/bin/llama-server"
}
EOF

# 3. 运行
python3 gui.py
```

**macOS 桌面快捷方式**：`~/Desktop/llama-shell.app` (原生 `.app` bundle，双击即启)

---

## 📋 界面一览

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

## ⚙️ 关键参数速查 (8GB VRAM + 32GB RAM 基线)

| 参数 | 位置 | 推荐值 (MoE) | 推荐值 (Dense) |
|------|------|--------------|----------------|
| `ngl` | ① | `999` (全卸载) | `auto` |
| `n-cpu-moe` | ① | `32` (35B) | — |
| `moe-cache-mib` | ① | `4096`~`8192` | — |
| `moe-cache-cpu-share` | ① | `0.5` | — |
| `dynamic-moe` | ① | `on` (Tosh) | — |
| `ctx` | ② | `65536` | `65536` |
| `cache-type-k/v` | ② | `q8_0` | `q8_0` |
| `flash-attn` | ② | `on` | `on` |
| `batch/ubatch` | ② | `4096/1024` | `2048/512` |
| `temp / top_p` | ③ | `0.75 / 0.95` | `0.75 / 0.95` |
| `repeat-penalty` | ③ | `1.1` | `1.1` |
| `presence-penalty` | ③ | `0.05` | `0.05` |

---

## 📁 配置文件

| 文件 | 作用 |
|------|------|
| `llama-shell-config.json` | 全局路径 (模型库、两引擎二进制) |
| `gui_state.json` | 按模型持久化 (参数、override、引擎、端口、Host) |
| `gui_presets.json` | 跨模型预设 (可导出/导入/另存为) |

---

## 🧪 质量保证

```bash
# 全部通过 ✅
python3 test_state.py      # 持久化 E2E (5 场景)
python3 test_prefill.py    # 推荐器 prefill (MoE + Dense)
python3 test_preset.py     # 预设/导出/导入 (7 场景)
```

---

## 🐛 已知问题 (v0.1.0)

- [ ] macOS Aqua 主题下 Canvas 绘制的运行/停止按钮文字极淡
- [ ] 思考模式按钮尚未统一为 "Thinking Mode" 风格三态开关
- [ ] 日志区建议再增 10 行 (现 32 行)
- [ ] 窗口整体高度可再压缩 ~50px (现 954px)
- [ ] MTP 推测解码实测效果待验证
- [ ] Windows 原生打包 (pyinstaller / msix) 待补充
- [ ] Linux / Wayland 适配待测试

---

## 🗺 路线图 (v0.2+)

- [ ] 按钮主题深度适配 (macOS / Windows / Linux 统一外观)
- [ ] 思考模式按钮重构 (参考主流 LLM 客户端交互)
- [ ] 基准测试内置 (自动跑 tok/s、首字延迟、显存占用)
- [ ] 模型下载/管理集成 (Hugging Face Hub / ModelScope)
- [ ] 多会话标签页 (同时跑多模型、多引擎对比)
- [ ] 插件系统 (自定义参数模板、导出格式、启动钩子)

---

## 📄 许可证

MIT License — 代码完全开源，二进制 `llama.cpp` / `ToshLLM` 遵循各自上游协议。

---

## 🙏 致谢

- **llama.cpp** (ggerganov) — 本地推理基石
- **ToshLLM** — Apple Silicon / AMD eGPU 深度优化
- **社区经验** — AMD RX 6600 XT 8GB VRAM + 32GB RAM 实测参数总结

---

> **一句话定位**：给「想跑大模型、不想折腾参数、要双引擎对比、要配置落盘」的开发者/研究者用的 **本地轻量控制台**。