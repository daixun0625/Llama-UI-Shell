#!/usr/bin/env python3
"""无 Tk 验证：prefill 预填 → 面板 dict → build_command 是否带上 -ngl/--n-cpu-moe。"""
import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recommend
import gguf_fast

M = "/Volumes/CT-nvme/llama-bin-win-vulkan-x64/models"
# 找 35B MoE
target = None
for p in sorted(os.listdir(M)):
    if p.endswith(".gguf") and "35B" in p and ("Abliterated" in p or "A3B" in p or "MoE" in p.upper()):
        target = os.path.join(M, p)
        break
if not target:
    for p in sorted(os.listdir(M)):
        if p.endswith(".gguf") and "mmproj" not in p:
            target = os.path.join(M, p)
            break
print("model:", target)

hw = recommend.detect_hardware()
print(f"hw: ram={hw.ram_gb:.0f}g vram={hw.vram_gb:.0f}g phys_cores={getattr(hw,'physical_cores','?')} "
      f"has_gpu={hw.has_gpu} metal_compute={hw.metal_compute}")

h = gguf_fast.parse_gguf_header(target)
st = recommend.model_stats(h)
print(f"st: {st['name']} total={st['total_b']:.0f}B is_moe={st.get('is_moe')} layers={st.get('layers')}")

# 面板全局默认（与 gui.py PARAM_DEFS 一致）
DEFAULTS = {
    "threads": "auto", "cpu_moe": "off", "ncpu_moe": "auto", "ngl": "auto",
    "ncpu_ffn": "auto", "moe_cache_mib": "0", "moe_cache_cpu_share": "auto",
    "dynamic_moe": "off", "swa_full": "off", "load_mode": "auto", "numa": "none",
    "ctx": "65536", "cache_type": "q8_0", "cache_reuse": "0", "flash_attn": "auto",
    "kv_offload": "on", "batch": "2048", "ubatch": "512", "cont_batching": "on",
    "prio": "normal", "poll": "50", "warmup": "off",
    "temp": "0.75", "top_k": "40", "top_p": "0.95", "min_p": "0.05",
    "repeat_penalty": "1.1", "presence_penalty": "0.05", "frequency_penalty": "0",
    "seed": "-1", "reasoning": "auto", "reasoning_effort": "default",
    "reasoning_format": "auto", "dynamic_moe": "off",
}

pre = recommend.prefill(st, hw)
print("\npre:", json.dumps(pre, ensure_ascii=False))

panel = dict(DEFAULTS)
panel.update({k: str(v) for k, v in pre.items()})

# plan：和 _refresh_recommend 一样用面板的 ctx/cache_type/ncmoe
ctx = int(panel["ctx"])
kv = panel["cache_type"]
ncmoe = int(panel["ncpu_moe"]) if panel["ncpu_moe"].isdigit() else None
plan = recommend.recommend(st, hw, ctx=ctx, kv_type=kv, ncmoe=ncmoe)
print("plan level:", plan.get("level"), "mode:", plan.get("mode"))

bin_path = recommend.VANILLA_BIN
cmd = recommend.build_command(st, hw, plan, bin_path, port=8080, extra=[],
                              params=panel, override_args="")
print("\nCMD:", " ".join(cmd))

# 断言
joined = " ".join(cmd)
assert "-ngl" in joined, "缺少 -ngl"
if st.get("is_moe"):
    # MoE: prefill 给 ngl=999；11-13 起 --n-cpu-moe 不再自动注入（OOM 降级为提示）
    assert "999" in joined, "MoE 期望 -ngl 999"
    assert "--n-cpu-moe" not in joined, "11-13 定规：不自动注入 ncmoe"
    if plan.get("oom_risk"):
        assert any("--n-cpu-moe" in h for h in (plan.get("hints") or [])), \
            "oom_risk 提示应含 --n-cpu-moe 建议值"
print("\nPASS: ngl 调整功能 + 自动预填管线正常")
