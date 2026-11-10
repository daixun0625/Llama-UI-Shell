# Llama UI Shell — 硬件感知推荐引擎 (~34KB)
# prefill(st, hw) 核心算法：
# 1. 硬件预算 (vram_budget = vram-1.0g, ram_budget = 32*0.72-2.0)
# 2. CTx 档降序试装 [65536, 32768, 16384, 8192, 4096]
# 3. MoE 分支：-ngl 999 + --n-cpu-moe N (ncmoe 按 VRAM 切分)
# 4. 稠密分支：全层 GPU 或纯 CPU (按 RAM 预算)
# 5. 返回面板键值字典，GUI _apply_prefill() 直接叠加
# FLAG_MAP: 支持 Vanilla/ToshLLM 双引擎标志映射
