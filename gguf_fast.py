# Llama UI Shell — GGUF 快速解析 (~20KB)
# 混合层识别：kv_layers (full-attn) / ssm_layers (SSM)
# KV 仅按 full-attn 层计算，SSM state 估算
# 支持 qwen35moe / qwen25 / llama 等架构
# 返回 total_layers, expert_bytes, other_gb, is_moe
