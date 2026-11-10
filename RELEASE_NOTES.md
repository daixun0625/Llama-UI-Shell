# Llama UI Shell — Release Notes

## v0.1 (2026-11-10)

### 新增功能
- **模型库浏览**：支持可视化浏览本地 GGUF 模型，点击即加载
- **硬件感知推荐**：自动检测显存/CPU/RAM，一键推荐最优参数
- **双引擎支持**：llama.cpp Vanilla (b12212) 与 ToshLLM 双引擎一键切换
- **配置持久化**：按模型独立保存参数，下次打开自动恢复
- **预设管理**：支持配置另存、导出、导入，方便分享与迁移

### 参数面板
- **列 ①**：CPU/MoE 核心参数（ngl, 线程数, 专家缓存等）
- **列 ②**：缓存与解码参数（KV 量化, 推测解码, MTP）
- **列 ③**：采样参数（温度, top-k, top-p, 重复惩罚等）

### 技术细节
- 纯 Tkinter GUI，零第三方依赖
- 自动显存管理，防止 OOM
- AMD eGPU macOS Metal 优化
- 端口自动清理，防止冲突

### 已知限制
- AMD RX 6600 XT 8GB VRAM 下 MoE 模型建议 ncmoe=32
- ToshLLM 在 Apple Silicon 上表现更佳

---

*本 UI 推荐 MOE 模型试用 ToshLLM 引擎，并感谢 TA。*