#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend.py — 推理参数推荐引擎（llama-shell macOS 版）

移植自 toshllm 0.87.11 的规划启发式（Engelbert Delgado），针对
本地硬件（i5-12400F / 32GB / macOS x64 + llama.cpp b12212）裁剪。

公式（与 toshllm 一致，保留原常数）：
  computeGB = 0.9 + paramsB * 0.012
  dense 全 GPU：need = weights*1.03 + kv + compute
  MoE：denseGB(attention+shared) 常驻 GPU；experts 按 VRAM 余量切分；
       ncmoe = ceil(layers * cpu_experts/experts)
  KV 精确值 = attn_layers * heads_kv * (klen+vlen) * 2 字节/token（f16）
  kvTypeScale: f16=1.0 q8_0=0.53 q5_1=0.38 q5_0=0.36 q4_1=0.32 其它≈0.30
  RAM 带宽按 48 GB/s；MoE 效率 0.48
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

GiB = 1024 ** 3

# 项目根目录（兼容开发环境与打包后 .app 内运行）
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 硬件探测

@dataclass
class Hardware:
    ram_gb: float
    physical_cores: int
    logical_cores: int
    cpu_brand: str
    gpus: list = field(default_factory=list)   # [{name, vram_gb, integrated}]
    ram_bw_gbps: float = 48.0                  # toshllm 常数
    metal_compute: bool = True                 # 是否有 Metal 可算 GPU
    vulkan_compute: bool = False               # 若走 Vulkan 后端，独显即可用作计算

    @property
    def best_gpu(self):
        dis = [g for g in self.gpus if not g.get("integrated")]
        return (dis or self.gpus or [None])[0]

    @property
    def vram_gb(self) -> float:
        return (self.best_gpu or {}).get("vram_gb", 0.0) or 0.0

    @property
    def has_gpu(self) -> bool:
        """可 offload 的 GPU：须真有 VRAM 且 Metal 可算（显示-only 不算）。"""
        g = self.best_gpu
        if not g or g.get("vram_gb", 0) <= 0.2:
            return False
        if not g.get("usable_compute", self.metal_compute):
            return False
        return True


def detect_hardware(ram_bw_override: float | None = None) -> Hardware:
    def sysctl(name: str):
        try:
            out = subprocess.run(["sysctl", "-n", name],
                                 capture_output=True, text=True, timeout=5)
            return out.stdout.strip()
        except Exception:
            return ""
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout or "0") / GiB
    pc = int(sysctl("hw.physicalcpu") or "0")
    lc = int(sysctl("hw.ncpu") or "0")
    brand = sysctl("machdep.cpu.brand_string")
    gpus: list[dict] = []
    try:
        prof = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                              capture_output=True, text=True, timeout=20).stdout
        data = json.loads(prof)
        for item in data.get("SPDisplaysDataType", []):
            name = item.get("_name") or item.get("sppci_model") or "?"
            mem = item.get("spdisplays_vram") or item.get("spdisplays_wired")
            vram = 0.0
            m = re.search(r"([\d.]+)\s*([GM])B", str(mem)) if mem else None
            if m:
                vram = float(m.group(1)) if m.group(2) == "G" else float(m.group(1)) / 1024.0
            low = name.lower()
            integrated = any(k in low for k in ("integrated", "uhd", "iris", "vga"))
            gpus.append({"name": name, "vram_gb": round(vram, 1),
                         "integrated": integrated})
    except Exception:
        pass
    for g in gpus:
        if g["integrated"]:
            g["vram_gb"] = min(g["vram_gb"], 0.5)   # iGPU 视为不可用预算
    hw = Hardware(ram_gb=ram, physical_cores=pc, logical_cores=lc,
                  cpu_brand=brand, gpus=gpus,
                  ram_bw_gbps=ram_bw_override or 48.0)
    hw.metal_compute = _metal_compute_available()
    for g in gpus:
        if g.get("integrated"):
            g["usable_compute"] = hw.metal_compute
        else:
            g["usable_compute"] = True
    hw.vulkan_compute = any((not g.get("integrated")) and g.get("vram_gb", 0) > 0.2
                            for g in gpus)
    return hw


def _metal_compute_available() -> bool:
    """本机是否存在 llama.cpp Metal 可识别的计算 GPU（非仅显示）。
    AMD 独显在 macOS 上走 Metal 3 驱动（非 Vulkan），系统级 Metal 支持即可。
    """
    # 方法 1：SPMetalDataType（Apple Silicon 或较新 Intel）
    try:
        out = subprocess.run(
            ["system_profiler", "SPMetalDataType"],
            capture_output=True, text=True, timeout=20).stdout
        if "Metal Devices" in out or ("GPU" in out and "VRAM" in out):
            return True
    except Exception:
        pass

    # 方法 2：SPDisplaysDataType 查 Metal Support 字段（AMD/NVIDIA 独显走 Metal 3）
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=20).stdout
        if "Metal Support" in out and "Metal" in out:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------- 模型参数解析

def _parse_size(name: str) -> tuple[float, float]:
    """从模型名解析 (总参数B, 激活参数B)。'A<N>B' 为 MoE 激活参数量。"""
    name = name.upper()
    active = 0.0
    am = re.search(r"A(\d+(?:\.\d+)?)\s?B", name)
    if am:
        active = float(am.group(1))
    tot = 0.0
    for mm in re.finditer(r"(\d+(?:\.\d+)?)\s?B", name):
        tot = max(tot, float(mm.group(1)))
    if active and not tot:
        tot = active
    return tot, active


# 量化 bytes/param（近似，用于由文件大小反推参数量）
QUANT_BYTES = {
    1: 2.0, 2: 0.56, 3: 0.56, 7: 1.06, 13: 0.44, 14: 0.55, 15: 0.57,
    16: 0.67, 17: 0.69, 18: 0.81, 31: 0.53, 32: 2.0, 38: 0.50, 39: 0.56,
    42: 1.0,
}


def model_stats(h) -> dict:
    """把 gguf_fast.ModelHeader 归纳成推荐引擎需要的量。"""
    file_gb = h.tensor_bytes / GiB
    expert_gb = h.expert_bytes / GiB
    layers = h.layers or 0
    is_moe = (h.expert_tensor_count > 0) or (h.expert_count > 0)
    raw_name = h.model_name or ""
    fallback = h.path.rsplit("/", 1)[-1] if getattr(h, "path", "") else "?"
    stem = fallback.rsplit(".", 1)[0]
    # 转换管线偶发把占位模板名写进 GGUF general.name（如 "Staged_Tmpl"）：
    # 头部名不含数字而文件名含 → 改用文件名 stem（B 数解析与显示随之修正）
    if raw_name and not any(c.isdigit() for c in raw_name) and any(c.isdigit() for c in stem):
        raw_name = stem
    name = raw_name or fallback
    tot, act = _parse_size(name)
    if not tot and file_gb > 0:
        bpp = QUANT_BYTES.get(h.file_type, 0.57)
        tot = file_gb * GiB / bpp / 1e9
    if is_moe and not act:
        act = tot * 0.11
    kv_bpt = getattr(h, "kv_bytes_per_token_f16", 0)
    ssm_state_gb = getattr(h, "ssm_state_bytes", 0) / GiB
    other_gb = h.other_bytes / GiB
    return {"name": name, "path": getattr(h, "path", ""), "file_gb": file_gb,
            "expert_gb": expert_gb, "other_gb": other_gb,
            "layers": layers, "is_moe": is_moe,
            "total_b": tot, "active_b": act, "kv_bpt": kv_bpt,
            "ssm_state_gb": ssm_state_gb,
            "kv_layers": getattr(h, "kv_layers", 0),
            "ssm_layers": getattr(h, "ssm_layers", 0),
            "expert_count": getattr(h, "expert_count", 0),
            "active_experts": getattr(h, "active_experts", 0),
            "mtp_layers": getattr(h, "mtp_layers", 0)}


# ---------------------------------------------------------------- 参数估算

KV_TYPE_SCALE = {"f16": 1.0, "q8_0": 0.53, "q5_1": 0.38, "q5_0": 0.36,
                 "q4_1": 0.32, "q4_0": 0.30, "iq4_nl": 0.30, "turbo4": 0.25}


def _kv_gb(st: dict, ctx: int, kv_type: str) -> float:
    """KV 缓存估算（f16 几何 × 量化折算）——estimate() 与 prefill() 共用。"""
    kv_bpt = st.get("kv_bpt")
    total_b = st.get("total_b", 0)
    if kv_bpt:
        kv_gb = kv_bpt * ctx / GiB
    else:
        per_k = 0.10 if total_b >= 25 else 0.08 if total_b >= 12 else 0.05
        kv_gb = per_k * ctx / 1024
    return kv_gb * KV_TYPE_SCALE.get(kv_type, 0.30)


def estimate(st: dict, ctx: int, kv_type: str, hw: Hardware,
             ncmoe_override: int | None = None,
             vram_reserve_gb: float = 1.0) -> dict:
    file_gb, expert_gb = st["file_gb"], st["expert_gb"]
    layers, is_moe, total_b, active_b = st["layers"], st["is_moe"], st["total_b"], st["active_b"]
    kv_bpt = st["kv_bpt"]

    vram_budget = max(0.0, hw.vram_gb - vram_reserve_gb)
    bw_vram = 384.0 if hw.has_gpu else hw.ram_bw_gbps   # 有显存按 HBM/独显带宽

    kv_gb = _kv_gb(st, ctx, kv_type)
    compute_gb = 0.9 + total_b * 0.012

    out: dict = {"ctx": ctx, "kv_type": kv_type, "kv_gb": round(kv_gb, 2),
                 "vram_budget_gb": round(vram_budget, 2), "warnings": []}

    if not is_moe:
        need = file_gb * 1.03 + kv_gb + compute_gb
        if hw.has_gpu and need <= vram_budget:
            out.update(mode="full_gpu", n_gpu_layers=layers, n_cpu_moe=0,
                       vram_need_gb=round(need, 2), ram_need_gb=0.7, level="ideal",
                       speed=f"~{max(4, int(bw_vram / max(0.5, file_gb)))} t/s")
            return out
        if hw.has_gpu and file_gb * 0.5 <= vram_budget and file_gb < hw.ram_gb * 0.6:
            on_gpu = max(0.0, vram_budget - kv_gb - compute_gb)
            frac_ram = min(1.0, max(0.0, (file_gb - on_gpu) / max(0.5, file_gb)))
            per_tok = file_gb * ((1 - frac_ram) / bw_vram + frac_ram / hw.ram_bw_gbps) / GiB
            out.update(mode="cpu_partial", n_gpu_layers=-1, n_cpu_moe=0,
                       vram_need_gb=round(vram_budget, 2),
                       ram_need_gb=round(file_gb - on_gpu + 2, 1),
                       level="slow",
                       speed=f"~{max(4, int(1 / max(1e-4, per_tok)))} t/s (CPU 主导)")
            return out
        if hw.has_gpu:
            out.update(mode="reject_gpu", n_gpu_layers=0, n_cpu_moe=0,
                       vram_need_gb=round(need, 2), level="no",
                       speed="纯 CPU 运行")
            out["hints"] = [
                "dense 模型超显存：可手动试 attention/GPU + FFN/CPU 切分，",
                "如 --n-cpu-ffn 40（配合 -ngl 调高，面板「n-cpu-ffn」项）"]
            out["warnings"].append(f"GPU 余量 {vram_budget:.1f} GiB 不够，转纯 CPU")
            return out
        ram_need = file_gb + 1.5
        if ram_need > hw.ram_gb * 0.72:
            out.update(mode="reject", level="no",
                       warnings=out["warnings"] + [f"需 ~{ram_need:.1f} GiB RAM（>72% 的 {hw.ram_gb:.0f} GiB）"])
            return out
        out.update(mode="cpu_only", n_gpu_layers=0, n_cpu_moe=0,
                   vram_need_gb=0, ram_need_gb=round(ram_need, 1), level="slow",
                   speed=f"~{max(4, int(hw.ram_bw_gbps / max(0.5, file_gb)))} t/s (CPU 主导)")
        return out

    # ---- MoE：attention/shared 常驻，experts 按余量切分
    dense_gb = max(0.3, file_gb - expert_gb) if expert_gb > 0 else 1.3
    ssm_state_gb = st.get("ssm_state_gb", 0.0)
    overhead_gb = dense_gb + 0.1 + kv_gb + compute_gb + ssm_state_gb
    experts = expert_gb if expert_gb > 0 else max(0.0, file_gb - 1.3)
    gpu_exp = min(experts, max(0.0, vram_budget - overhead_gb)) if hw.has_gpu else 0.0
    cpu_exp = experts - gpu_exp
    ram_need = cpu_exp + 1.0

    if hw.has_gpu:
        if ram_need > hw.ram_gb * 0.72:
            out.update(mode="reject", level="no",
                       warnings=out["warnings"] + [f"需 ~{ram_need:.1f} GiB 主机内存（>72%）"])
            return out
        auto_ncmoe = min(layers, int(layers * (cpu_exp / experts) + 0.5)) if experts else 0
        warns = list(out["warnings"])
        if ncmoe_override is not None:
            ncmoe = max(0, min(layers, ncmoe_override))
        else:
            ncmoe = auto_ncmoe
            while ncmoe < layers and (overhead_gb + experts * (1 - ncmoe / max(1, layers))) > vram_budget + 1e-6:
                ncmoe += 1
        frac = min(1.0, ncmoe / max(1, layers))
        c_exp = experts * frac
        g_exp = experts - c_exp
        vram_need_now = overhead_gb + g_exp
        if ncmoe_override is not None and vram_need_now > vram_budget + 1e-6:
            warns.append(f"强制 {ncmoe} 层进 CPU：vram_need {vram_need_now:.2f}g 超预算 {vram_budget:.1f}g，可能 OOM/换页")
        bps = file_gb / max(1.0, total_b)
        act_gb = active_b * bps
        per_tok = act_gb * ((1 - frac) / bw_vram + frac / hw.ram_bw_gbps) / GiB
        tg = 0.48 / max(1e-4, per_tok)
        tg = min(max(60.0, 384.0 / 4), max(4.0, tg))
        lvl = "slow" if frac > 0.85 else "good"
        if vram_need_now > vram_budget + 1e-6:
            lvl = "warn"
        if ncmoe > 30:
            warns.append(f"{ncmoe} 层专家在 CPU：prefill 收益低于成本，pp 会明显变慢")
        hints = []
        free_gb = vram_budget - vram_need_now
        if ncmoe > 0 and free_gb >= 0.5:
            hints.append(
                f"experts 在 CPU 且 VRAM 余 ~{free_gb:.1f}g：vanilla 可加 "
                f"--moe-cache-mib ≤{max(256, int(free_gb * 1024))}（热专家上 GPU，提速但吃显存）")
        elif ncmoe > 0:
            hints.append("VRAM 已满，勿再加 --moe-cache-mib（会挤爆 8G）")
        # 11-13 定规：不再自动注入 --n-cpu-moe；全专家上显存的 OOM 风险降级为提示
        # （GUI info 红字），用户需要时把建议值复制进自定义参数框
        full_gpu_gb = overhead_gb + g_exp + c_exp
        oom_risk = ncmoe > 0 and full_gpu_gb > vram_budget + 1e-6
        if oom_risk:
            hints.append(
                f"⚠️ OOM 风险：模型+全部专家 ~{full_gpu_gb:.1f}g 装不下 {vram_budget:.0f}g 显存，"
                f"-ngl 999 不配 --n-cpu-moe 会加载失败；建议复制进自定义参数框：--n-cpu-moe {ncmoe}")
        out.update(mode="moe_split", n_gpu_layers=-1, n_cpu_moe=ncmoe, oom_risk=oom_risk,
                   vram_need_gb=round(overhead_gb + g_exp, 2),
                   ram_need_gb=round(c_exp + 1.0, 1), level=lvl,
                   speed=f"~{max(4, int(tg * 0.8))}-{int(tg)} t/s",
                   gpu_experts_gb=round(g_exp, 1), cpu_experts_gb=round(c_exp, 1),
                   hints=hints,
                   warnings=warns)
        return out

    # 无 GPU 的 MoE：experts 全走 CPU
    if ram_need > hw.ram_gb * 0.72:
        out.update(mode="reject", level="no",
                   warnings=out["warnings"] + [f"需 ~{ram_need:.1f} GiB 主机内存（>72% 的 {hw.ram_gb:.0f}）"])
        return out
    act_gb = active_b * (file_gb / max(1.0, total_b))
    out.update(mode="cpu_only", n_gpu_layers=0, n_cpu_moe=0,
               vram_need_gb=0, ram_need_gb=round(ram_need, 1), level="slow",
               speed=f"~{max(4, int(hw.ram_bw_gbps / max(0.5, act_gb)))} t/s (CPU，active 参数)")
    return out


# ---------------------------------------------------------------- 上下文推荐

def recommend(st: dict, hw: Hardware, ctx: int | None = None,
              kv_type: str | None = None, ncmoe: int | None = None) -> dict:
    """主入口：自动选 ctx/KV，输出完整推荐。st = model_stats(h)。"""
    ctx = ctx or 16384
    kv = kv_type or "q8_0"  # 默认档位 q8_0（显存友好、精度损失极小）
    trial = estimate(st, ctx, kv, hw, ncmoe_override=ncmoe)
    if trial.get("vram_need_gb", 0) > trial.get("vram_budget_gb", 1e9) * 0.98 and kv == "f16":
        trial2 = estimate(st, ctx, "q8_0", hw, ncmoe_override=ncmoe)
        if trial2.get("ram_need_gb", 0) <= trial.get("ram_need_gb", 1e9) + 2:
            trial = trial2
            trial["warnings"] = ["KV 降为 q8_0 以保证内存余量（精度损失很小）"] + list(trial2.get("warnings", []))
    trial["model"] = st["name"]
    return trial


# 各参数 key -> flag 映射（flag 名与取值均已从 b12212 两个二进制的 --help 实测确认）
# 值为 None = 省略该 flag（保持二进制默认）
# "default" / "" / "不配置" 在 build_command 统一跳过
FLAG_MAP = {
    # CPU / MoE
    "threads": lambda v: ["--threads", str(v)] if str(v).strip() not in ("", "auto", "default", "不配置") else None,
    "cpu_moe": lambda v: ["--cpu-moe"] if str(v).strip() == "on" else None,
    "moe_cache_mib": lambda v: ["--moe-cache-mib", str(v)] if str(v).strip() not in ("0", "", "default", "不配置") else None,
    "moe_cache_cpu_share": lambda v: ["--moe-cache-cpu-share", str(v)] if str(v).strip() not in ("0", "0.0", "", "default", "不配置") else None,
    # 直传型数值参数：选 "default"/"不配置"/空 时同样必须省略 flag（llama-server 用自身默认值），
    # 否则命令行出现字面 "default"（如 --ctx-size default），arg 解析直接失败→启动失败（11-13 主人报告）
    "batch": lambda v: ["--batch-size", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "ubatch": lambda v: ["--ubatch-size", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "no_warmup": lambda v: ["--no-warmup"] if str(v).strip() == "on" else None,
    "dynamic_moe": lambda v: ["--dynamic-moe", "on"] if str(v).strip() == "on" else None,
    # 上下文 / KV / FA / load
    "ctx": lambda v: ["--ctx-size", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "flash_attn": lambda v: ["--flash-attn", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    # K/V 分开设（b12212 实测 -ctk/-ctv）；GUI 旧键 cache_type 由 GUI 层迁移到 k/v
    "cache_type_k": lambda v: ["--cache-type-k", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "cache_type_v": lambda v: ["--cache-type-v", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    # MTP（Multi-Token Prediction）投机解码，b12212 vanilla 实测支持
    "spec_type": lambda v: ["--spec-type", str(v)] if str(v).strip() not in ("", "off", "none", "default", "不配置") else None,
    "spec_draft_model": lambda v: ["--spec-draft-model", str(v)] if str(v).strip() not in ("", "off", "auto", "default", "不配置") else None,
    "spec_draft_n_max": lambda v: ["--spec-draft-n-max", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "spec_draft_cache_type_k": lambda v: ["--cache-type-k-draft", str(v)] if str(v).strip() not in ("", "auto", "default", "不配置") else None,
    "spec_draft_cache_type_v": lambda v: ["--cache-type-v-draft", str(v)] if str(v).strip() not in ("", "auto", "default", "不配置") else None,
    "cache_reuse": lambda v: ["--cache-reuse", str(v)] if str(v).strip() not in ("0", "", "default", "不配置") else None,
    # cache-ram：KV 缓存 RAM 上限（MiB，1k 分档）；default/0=省略（二进制默认 8192）
    "cache_ram": lambda v: ["--cache-ram", str(v)] if str(v).strip() not in ("", "0", "default", "不配置") else None,
    "swa_full": lambda v: ["--swa-full"] if str(v).strip() == "on" else None,
    "load_mode": lambda v: ["--load-mode", str(v)] if str(v).strip() not in ("", "auto", "default", "不配置") else None,
    # 多模态 / 视觉塔（mmproj）
    "mmproj": lambda v: ["--mmproj", str(v)] if str(v).strip() not in ("", "off", "auto", "default", "不配置") else None,
    "mmproj_device": lambda v: ["--mmproj-device", str(v)] if str(v).strip() not in ("none", "auto", "", "default", "不配置") else None,
    "image_min_tokens": lambda v: ["--image-min-tokens", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "image_max_tokens": lambda v: ["--image-max-tokens", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    # agent / jinja
    "agent": lambda v: ["--agent"] if str(v).strip() == "on" else None,
    "jinja": lambda v: ["--jinja"] if str(v).strip() == "on" else None,
    # 采样 / 惩罚 / 思考
    "temp": lambda v: ["--temp", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "top_k": lambda v: ["--top-k", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "top_p": lambda v: ["--top-p", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "min_p": lambda v: ["--min-p", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "repeat_penalty": lambda v: ["--repeat-penalty", str(v)] if str(v).strip() not in ("1.0", "1.00", "", "default", "不配置") else None,
    # repeat-last-n：惩罚回看窗口（token 数，二进制默认 64）；default/64=省略
    "repeat_last_n": lambda v: ["--repeat-last-n", str(v)] if str(v).strip() not in ("", "64", "default", "不配置") else None,
    "presence_penalty": lambda v: ["--presence-penalty", str(v)] if str(v).strip() not in ("0", "0.0", "", "default", "不配置") else None,
    "frequency_penalty": lambda v: ["--frequency-penalty", str(v)] if str(v).strip() not in ("0", "0.0", "", "default", "不配置") else None,
    "seed": lambda v: ["--seed", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    # 思考模式（Qwen3 系 thinking）
    "reasoning": lambda v: ["--reasoning", str(v)] if str(v).strip() not in ("", "auto", "default", "不配置") else None,
    "reasoning_effort": lambda v: ["--reasoning-effort", str(v)] if str(v).strip() not in ("", "default", "不配置") else None,
    "reasoning_format": lambda v: ["--reasoning-format", str(v)] if str(v).strip() not in ("", "auto", "default", "不配置") else None,
}

# 不带取值的长 flag（消解冲突时不吞下一个 token）
_BOOL_LONG_FLAGS = {"warmup", "no-warmup", "swa-full", "no-cont-batching",
                    "cont-batching", "no-kv-offload", "kv-offload",
                    "mmproj-offload", "no-mmproj-offload"}

# flag 别名族（冲突消解：用户自定义覆盖时按族去重；-c 与 --ctx-size 同族）
FLAG_ALIASES = [
    {"ctx-size", "c"}, {"threads", "t"}, {"batch-size", "b"}, {"ubatch-size", "ub"},
    {"cont-batching", "cb", "no-cont-batching", "nocb"}, {"prio"}, {"poll"},
    {"warmup", "no-warmup"}, {"dynamic-moe"}, {"flash-attn", "fa"},
    {"n-cpu-moe", "ncmoe", "cpu-moe", "cmoe"}, {"n-cpu-ffn", "ncffn"},
    {"moe-cache-mib"}, {"moe-cache-cpu-share"},
    {"cache-type-k", "ctk"}, {"cache-type-v", "ctv"}, {"cache-reuse"}, {"swa-full"},
    {"load-mode", "lm"}, {"numa"}, {"kv-offload", "kvo", "no-kv-offload", "nkvo"},
    {"temp", "temperature"}, {"top-k"}, {"top-p"}, {"min-p"},
    {"repeat-penalty", "rep-penalty"}, {"presence-penalty", "pre-penalty"},
    {"frequency-penalty", "freq-penalty"}, {"seed", "s"},
    {"reasoning", "rea"}, {"reasoning-effort"}, {"reasoning-format"},
    # MTP 投机解码族（b12212 vanilla）
    {"spec-type"}, {"spec-draft-model", "model-draft"},
    {"spec-draft-n-max", "spec-ngram-mod-n-max"},
    {"cache-type-k-draft", "ctkd"}, {"cache-type-v-draft", "ctvd"},
    # 多模态 / 视觉塔族
    {"mmproj"}, {"mmproj-offload", "mmproj-no-offload", "no-mmproj-offload"},
    {"mmproj-device"}, {"image-min-tokens"}, {"image-max-tokens"},
    # agent / jinja
    {"agent"}, {"jinja"},
]


def _flag_family(tok: str):
    """一个 flag token（'-c'/'--ctx-size'）所属别名族。"""
    bare = tok.lstrip("-")
    for fam in FLAG_ALIASES:
        if bare in fam:
            return frozenset(fam)
    return None


def _strip_flag_pairs(tokens: list, families: set) -> list:
    """删掉 tokens 中属于冲突族的 flag 及其取值（--flag value / -f value 形式）。"""
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        fam = _flag_family(tok) if tok.startswith("-") else None
        if fam and fam in families:
            take_value = (tok.startswith("--")
                          and tok.lstrip("-") not in _BOOL_LONG_FLAGS
                          and "=" not in tok)
            i += 2 if take_value else 1
            continue
        out.append(tok)
        i += 1
    return out


def merge_custom_overrides(cmd_tokens: list, override_args: str) -> tuple[list, list]:
    """把用户自定义参数字符串并入命令。同名 flag（含别名）以用户为准：
    先删掉生成侧冲突族的 flag，再把用户参数整体追加到末尾。
    返回 (新 token 列表, 被覆盖的冲突族可读列表)。"""
    import shlex
    user = shlex.split(override_args.strip() or "")
    if not user:
        return cmd_tokens, []
    families = set()
    for tok in user:
        if tok.startswith("-"):
            fam = _flag_family(tok)
            if fam:
                families.add(fam)
    if not families:
        return cmd_tokens + user, []
    base = _strip_flag_pairs(cmd_tokens, families)
    hit = sorted("或".join(sorted(f)) for f in families)
    return base + user, hit


def build_command(st: dict, hw: Hardware, plan: dict, binary: str,
                  port: int = 8080, host: str = "127.0.0.1", extra: list | None = None,
                  dynamic_moe: bool = False,
                  params: dict | None = None,
                  override_args: str = "",
                  clear_mode: str = "override") -> list:
    """把 plan 翻译成 llama-server 参数（本地自动探测最高 bin/llama-b*；llama-server / llama-cli 通用）；
    tosh 二进制缺 moe-cache-* 旗标时自动省略。
    clear_mode:
      - "override"（默认）：框架 + 面板参数（default 跳过）+ 用户 override（去重）
      - "clear"：仅框架（binary -m 模型 --host --port）+ 用户 override，**无任何自动参数**"""
    cmd = [binary, "-m", st["path"], "--host", host, "--port", str(port)]
    if clear_mode == "clear":
        # 清空模式：不加任何面板/自动参数，直接加用户自定义覆盖
        if override_args:
            import shlex
            cmd += shlex.split(override_args.strip())
        return cmd
    p = dict(params or {})
    cmoe_on = p.get("cpu_moe") == "on"
    # ngl：面板可显式指定（含预填的具体数字）；auto 时按 plan 走；default 不下发
    ngl_param = str(p.get("ngl", "auto")).strip()
    ngl_skip = ngl_param in ("", "default", "不配置")
    ngl_auto = ngl_param in ("", "auto", "default", "不配置", "None")
    ngl_val = plan.get("n_gpu_layers", 0)
    # 11-13 定规（主人）：面板无 n-cpu-moe 数字字段 → 一律不自动注入 --n-cpu-moe；
    # 命令只含面板有字段的参数。MoE 全专家上显存的 OOM 风险降级为 estimate() 的
    # oom_risk 提示（GUI info 红字），用户需要时把 `--n-cpu-moe N` 复制进自定义参数框。
    if ngl_skip:
        # 用户显式选 default → 完全不下发 -ngl（二进制默认 -1=尽量卸载）
        pass
    elif not ngl_auto:
        # 用户/预填显式指定 ngl（具体数字） → 直接采用（优先于 plan）
        cmd += ["-ngl", ngl_param]
    elif ngl_val > 0:
        cmd += ["-ngl", str(ngl_val)]
    elif ngl_val == -1:
        if st.get("is_moe"):
            # MoE 拆分：-ngl 999（全层上 GPU）；是否配 --n-cpu-moe 由用户显式决定
            cmd += ["-ngl", "999"]
        else:
            # dense cpu_partial：部分卸载（ngl=一半层），用户可再调
            half = int(st.get("layers", 0) // 2)
            if half > 0:
                cmd += ["-ngl", str(half)]
    else:
        # ngl_val == 0 → 全 CPU
        cmd += ["-ngl", "0"]
    # 面板参数（FLAG_MAP）：默认值省略；冲突族由 merge_custom_overrides 消解
    # 旧版面板键 cache_type 兼容：未拆 k/v 时映射到两个
    if "cache_type" in p and "cache_type_k" not in p and "cache_type_v" not in p:
        p = dict(p)
        p["cache_type_k"] = p.pop("cache_type")
        p["cache_type_v"] = p["cache_type_k"]
    spec_on = str(p.get("spec_type", "off")).strip() not in ("", "off", "none", "default", "不配置")
    is_tosh = "ToshLLM" in binary
    for key, val in p.items():
        if key == "dynamic_moe" and not is_tosh:
            continue
        if key in ("moe_cache_mib", "moe_cache_cpu_share") and is_tosh:
            continue  # ToshLLM 二进制无 moe-cache-* 旗标（help 实测）
        if key in ("spec_draft_model", "spec_draft_n_max") and not spec_on:
            continue  # MTP 关闭时不单独下发 draft 参数
        if key == "ncpu_moe" and cmoe_on:
            continue  # --cpu-moe 已含全部专家上 CPU，-ncmoe 冗余
        fn = FLAG_MAP.get(key)
        if fn is None:
            continue
        out = fn(str(val))
        if out:
            cmd += out
    if extra:
        cmd += extra
    cmd = merge_custom_overrides(cmd, override_args)[0]
    return cmd


def latest_vanilla_bin() -> str:
    """本地兜底：扫描 bin/llama-b*/llama-server，取最高 build 号。
    升级 llama 只需在 bin/ 放新的 llama-bXXXX/ 目录，无需改代码。"""
    bindir = os.path.join(BASE, "bin")
    best_num, best_path = 0, os.path.join(bindir, "llama-b12212", "llama-server")
    if os.path.isdir(bindir):
        for d in os.listdir(bindir):
            m = re.match(r"^llama-b(\d+)$", d)
            if m and os.path.isfile(os.path.join(bindir, d, "llama-server")):
                if int(m.group(1)) > best_num:
                    best_num, best_path = int(m.group(1)), os.path.join(bindir, d, "llama-server")
    return best_path


def _find_vanilla_bin() -> str:
    import shutil
    p = shutil.which("llama-server")
    if p:
        return p
    return latest_vanilla_bin()
VANILLA_BIN = _find_vanilla_bin()
TOSH_BIN = "/Applications/ToshLLM.app/Contents/Resources/bin/llama-server"


def pick_engine(st: dict, hw: Hardware, forced: str = "auto") -> tuple:
    """返回 (engine, binary, notes)。forced: auto/vanilla/tos（GUI 引擎单选）。
    实测（Gemma4-26B-A4B 纯 CPU）：ToshLLM 13.4 t/s > vanilla 9.3 t/s。
    - 有 Metal 计算 GPU → tosh（AutoMemoryPlan 最优切分）
    - MoE（纯 CPU 也快）→ tosh（其 CPU MoE 内核更快，dynamic-moe 默认 off）
    - 其余 → vanilla
    """
    tosh_ok = os.path.exists(TOSH_BIN)
    notes = []
    if st.get("is_moe"):
        engine = "tosh"
        reason = "MoE 模型 → ToshLLM（CPU MoE 内核实测更快）"
    elif hw.metal_compute:
        engine = "tosh"
        reason = "有 Metal 计算 GPU → ToshLLM（VRAM/CPU 最优切分）"
    else:
        engine = "vanilla"
        reason = "无 Metal 计算 GPU（显存仅显示）→ vanilla CPU 后端"
    notes.append(reason)
    if not tosh_ok:
        if engine == "tosh":
            notes.append("但 ToshLLM 二进制缺失 → 回退 vanilla")
            engine = "vanilla"
    if forced in ("vanilla", "tos"):
        want = "tosh" if forced == "tos" else "vanilla"
        if want != engine:
            notes.append(f"用户指定 → {want}")
            engine = want
            if want == "tosh" and not tosh_ok:
                notes.append("⚠ ToshLLM 二进制缺失")
    binary = binary_for(engine)
    if st.get("mmap_ok") is False:
        notes.append("模型在 NVMe/网络卷：mmap 会失败 → 整读 RAM（需 32GiB）")
    return engine, binary, notes


def binary_for(engine: str) -> str:
    """tosh/auto(且 ToshLLM 存在) → ToshLLM 二进制；否则本地最高编号 llama-b*（自动探测）。"""
    if engine in ("tosh", "auto") and os.path.exists(TOSH_BIN):
        return TOSH_BIN
    return VANILLA_BIN


def engine_binary(engine: str) -> str:
    return binary_for(engine)


# ---------------------------------------------------------------- GUI 预填（每模型自动优化默认值）

# ctx 上限：首选 65536（与用户默认一致），装不下就逐级降
CTX_PREFILL_CANDS = [65536, 32768, 16384, 8192, 4096]


def _moe_cache_mib_from_hints(hints) -> str:
    """从 hint 文本提取 --moe-cache-mib 上限（MiB）。无缓存余量则返回 ''。"""
    import re
    for h in hints or []:
        m = re.search(r"--moe-cache-mib\s*≤\s*(\d+)", h)
        if m:
            return m.group(1)
    return ""


def prefill(st: dict, hw: Hardware, kv_type: str = "q8_0") -> dict:
    """每模型自动优化预填值：尽量把模型/专家装进显存（提速），并取装得下的最大 ctx。
    返回面板 key 的 dict，GUI 在模型无持久记录时以它作 *可编辑* 默认值。

    社区经验来源（2026-11，RX 6600 8GB VRAM / macOS Metal）：
    - MoE (Qwen3-35B-A3B)：-ngl 999 + --n-cpu-moe 32 + q8_0 KV + fa=on
      → ToshLLM dynamic-moe ~24 T/s；vanilla + -b 4096 -ub 1024 → 8K prefill 222 tok/s
    - dense (Qwen2.5-27B Q4_K_M ~15.7 GiB)：8GB VRAM 装不下 → 纯 CPU，
      32GB RAM budget (72%−2G=21 GiB) 够 ctx=65536 q8_0
    - threads = 物理核数（i5-12400F → 6）比逻辑核更稳
    - MoE 大 ctx 开 flash-attn（省 KV 显存）；dense 小 ctx 交给 auto
    - MoE：-ngl 999 + --n-cpu-moe N；moe_cache_mib 有值时配 moe_cache_cpu_share=0.5
    - dense 全显存存得下 → -ngl=全部层数；存不下(cpu_partial) → -ngl=一半
    """
    chosen = None
    for c in CTX_PREFILL_CANDS:
        p = recommend(st, hw, ctx=c, kv_type=kv_type)
        if p.get("level") not in ("warn", "no"):
            chosen = (c, p)
            break
        chosen = (c, p)  # 回退：最小 ctx
    if chosen is None:
        c = CTX_PREFILL_CANDS[-1]
        chosen = (c, recommend(st, hw, ctx=c, kv_type=kv_type))
    c, p = chosen

    # dense 超显存（reject_gpu → 纯 CPU）：显存装不下，但主机内存往往还有余量
    # → 按 RAM 预算（72% − 2g 留白）重挑 ctx：取装得下的最大档，不白丢上下文
    if not st.get("is_moe") and p.get("mode") == "reject_gpu" and hw.has_gpu:
        ram_budget = hw.ram_gb * 0.72 - 2.0
        compute_gb = 0.9 + st.get("total_b", 0) * 0.012
        best = None
        for cand in CTX_PREFILL_CANDS:  # 已降序
            need = st["file_gb"] * 1.03 + _kv_gb(st, cand, kv_type) + compute_gb + 1.0
            if need <= ram_budget:
                best = cand
                break
        if best and best != c:
            c, p = best, recommend(st, hw, ctx=best, kv_type=kv_type)

    # 社区最佳（RX 6600 8GB）：MoE prefill-optimized -b 4096 -ub 1024；dense -b 2048 -ub 512
    _batch, _ubatch = ("4096", "1024") if st.get("is_moe") else ("2048", "512")
    pre = {
        "ctx": str(p.get("ctx", c)),
        "cache_type_k": p.get("kv_type", "q8_0"),
        "cache_type_v": p.get("kv_type", "q8_0"),
        "flash_attn": "on" if st.get("is_moe") else "auto",
        "threads": str(hw.physical_cores) if getattr(hw, "physical_cores", 0) else "auto",
        "batch": _batch,
        "ubatch": _ubatch,
        "cache_ram": "2048",
        "cpu_moe": "off",
        "agent": "on",
        "jinja": "on",
    }
    if st.get("is_moe"):
        pre["ngl"] = "999"
        # 11-5 起面板无 n-cpu-moe 数字参数；11-13 定规后 build_command 不再自动
        # 注入 --n-cpu-moe（OOM 风险见 estimate() 的 oom_risk 提示，用户手动加自定义框）
        mib = _moe_cache_mib_from_hints(p.get("hints"))
        if mib:
            pre["moe_cache_mib"] = mib
            # vanilla：配 moe-cache-cpu-share 0.5（一半专家走 CPU share）
            # ToshLLM 无此旗标，build_command 会自动省略
            pre["moe_cache_cpu_share"] = "0.5"
    else:
        ngl = p.get("n_gpu_layers", 0)
        if ngl > 0:
            pre["ngl"] = str(ngl)          # 全层进显存
        elif ngl == -1:
            pre["ngl"] = str(int(st.get("layers", 0) // 2))  # cpu_partial：一半进显存
        else:
            pre["ngl"] = "0"              # reject / cpu_only：全 CPU
    return pre


if __name__ == "__main__":
    import gguf_fast, os, glob
    M = "/Volumes/CT-nvme/llama-bin-win-vulkan-x64/models"
    hw = detect_hardware()
    tosh_ok = os.path.exists(TOSH_BIN)
    print(f"HW: {hw.cpu_brand} | {hw.ram_gb:.0f}GiB | "
          f"{hw.physical_cores}C/{hw.logical_cores}T | Metal计算={hw.metal_compute} | GPUs: "
          + (", ".join(f"{g['name']} {g['vram_gb']}GiB{'(iGPU)' if g['integrated'] else ''}"
                       for g in hw.gpus) or "none"))
    print(f"Engines: vanilla={'OK' if os.path.exists(VANILLA_BIN) else '缺失'} ({os.path.basename(os.path.dirname(VANILLA_BIN))}) "
          f"tosh={'OK' if tosh_ok else '缺失'}")
    for p in sorted(glob.glob(M + "/*.gguf")):
        if "mmproj" in p or "clip" in p or "whisper" in p.lower():
            continue
        h = gguf_fast.parse_gguf_header(p)
        st = model_stats(h)
        plan = recommend(st, hw)
        engine, bin_path, eng_notes = pick_engine(st, hw)
        p = {"threads": "auto", "temp": "0.8", "top_p": "0.95",
             "top_k": "40", "flash_attn": "on", "dynamic_moe": "off"}
        cmd = build_command(st, hw, plan, bin_path, params=p)
        print(f"\n== {st['name']}  ({st['total_b']:.0f}B"
              + (f" MoE act={st['active_b']:.0f} exp={st['expert_count']}" if st["is_moe"] else " dense")
              + f")  engine={engine}")
        print(f"   {'; '.join(eng_notes)}")
        print(f"   mode={plan['mode']} level={plan['level']} ctx={plan['ctx']} kv={plan['kv_type']}")
        print(f"   ngl={plan.get('n_gpu_layers')} ncmoe={plan.get('n_cpu_moe')} "
              f"vram_need={plan.get('vram_need_gb')} ram_need={plan.get('ram_need_gb')}")
        print(f"   {plan.get('speed', '')}")
        if plan["warnings"]:
            print("   ! " + "; ".join(plan["warnings"]))
        print("   CMD: " + " ".join(cmd))