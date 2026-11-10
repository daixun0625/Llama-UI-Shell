# Llama UI Shell

> 轻量级 Tkinter GUI，封装 `llama.cpp` (b12212) 与 **ToshLLM**，专为本地大模型推理设计。

## 核心功能
- **模型库浏览**：可视化浏览本地 GGUF 模型库
- **硬件感知**：自动检测显存/CPU/RAM，推荐最优参数
- **双引擎**：支持 llama.cpp Vanilla 与 ToshLLM 双引擎切换
- **一键运行**：参数面板实时调整，启动即跑
- **持久化**：按模型独立保存参数设置
- **预设管理**：支持配置另存、导出、导入

## 硬件基线
- **CPU**：Intel i5-12400F (6C/12T)
- **RAM**：32 GB DDR4
- **GPU**：AMD Radeon RX 6600 XT 8GB VRAM

## 实测跑分
| 引擎 | 35B MoE 速度 | 备注 |
|------|--------------|------|
| **ToshLLM** | **~24 tok/s** | 硬件感知自动规划 |
| Vanilla b12212 | ~14 tok/s | 基线 |

## 文件说明
- `gui.py` — 主界面
- `recommend.py` — 硬件感知推荐引擎
- `gguf_fast.py` — GGUF 快速解析
- `llama-shell-config.json` — 配置文件

## License
MIT