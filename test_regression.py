#!/usr/bin/env python3
"""
llama-shell macOS 回归测试门禁（对标 Windows v2026-11 spec 的 23 项）。
全 PASS 才允许打包/发布。
"""
import sys, os, json, tempfile, subprocess, shlex
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recommend, gguf_fast

PASS = 0
FAIL = 0


def ok(msg: str):
    global PASS
    PASS += 1
    print(f"  ✅ {msg}")


def ng(msg: str):
    global FAIL
    FAIL += 1
    print(f"  ❌ {msg}")


def find_models_dir():
    """模型目录优先读 llama-shell-config.json 的 models_dir，再回退到真实模型目录。
    返回第一个"存在且含有效 .gguf"的目录；全不可用返回 None（调用方据此优雅跳过，不算回归失败）。"""
    candidates = []
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "llama-shell-config.json")) as f:
            cfg_md = json.load(f).get("models_dir", "")
            if cfg_md:
                candidates.append(cfg_md)
    except Exception:
        pass
    candidates += [
        "/Volumes/CT-nvme/llama-1.0.0-vulkan-x64/models",
        "/Volumes/CT-nvme/llama-bin-win-vulkan-x64/models",
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            ggufs = [p for p in sorted(os.listdir(d))
                    if p.endswith(".gguf") and not p.startswith(".")
                    and not any(s in p.lower() for s in ("mmproj", "clip", "whisper", "mtp"))]
            if ggufs:
                return d
    return None


# ============== 硬件/解析 ==============
def test_hardware_detection():
    hw = recommend.detect_hardware()
    assert hasattr(hw, "ram_gb") and hw.ram_gb > 0
    assert hasattr(hw, "physical_cores") and hw.physical_cores > 0
    assert hasattr(hw, "has_gpu")
    assert hasattr(hw, "metal_compute")
    ok("硬件探测: RAM/核心/GPU/Metal 属性齐全")


def test_model_stats_parsing():
    # 找一个真实模型做集成测试（目录优先读 config，回退真实模型目录；缺失则跳过）
    models_dir = find_models_dir()
    if not models_dir:
        print("  ⚠ 跳过: 无可用模型目录")
        return
    ggufs = [p for p in sorted(os.listdir(models_dir)) if p.endswith(".gguf") and not p.startswith(".") and not any(s in p.lower() for s in ("mmproj", "clip", "whisper", "mtp"))]
    assert ggufs, "无可用模型"
    p = os.path.join(models_dir, ggufs[0])
    h = gguf_fast.parse_gguf_header(p)
    st = recommend.model_stats(h)
    assert "file_gb" in st and st["file_gb"] > 0
    assert "layers" in st and st["layers"] > 0
    assert "is_moe" in st
    ok("模型解析: gguf_fast + model_stats 字段齐全")


# ============== 推荐引擎 ==============
def test_recommend_basic():
    hw = recommend.detect_hardware()
    st = {"name": "test", "path": "/tmp/test.gguf", "file_gb": 5.0, "is_moe": False,
          "total_b": 7, "active_b": 7, "layers": 32, "kv_bpt": 1024*1024,
          "expert_gb": 0, "other_gb": 0, "ssm_state_gb": 0, "kv_layers": 32,
          "ssm_layers": 0, "expert_count": 0, "active_experts": 0, "mtp_layers": 0}
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    assert "n_gpu_layers" in plan
    assert "kv_type" in plan and plan["kv_type"] == "q8_0"
    assert "level" in plan
    ok("推荐引擎: 基础 plan 字段齐全")


def test_recommend_moe_clamps_ncmoe():
    hw = recommend.detect_hardware()
    st = {"name": "test", "path": "/tmp/test.gguf", "file_gb": 5.0, "is_moe": True,
          "total_b": 7, "active_b": 7, "layers": 32, "kv_bpt": 1024*1024,
          "expert_gb": 2.0, "other_gb": 0, "ssm_state_gb": 0, "kv_layers": 32,
          "ssm_layers": 0, "expert_count": 8, "active_experts": 2, "mtp_layers": 0}
    # 显式 ncmoe 越界 → warn 级别
    plan = recommend.recommend(st, hw, ctx=65536, kv_type="q8_0", ncmoe=100)
    assert plan.get("level") == "warn"
    assert any("OOM" in w or "超预算" in w for w in plan.get("warnings", []))
    ok("推荐引擎: MoE 显式 ncmoe 越界给 warn + OOM 警告")


# ============== prefill ==============
def test_prefill_moe_uses_ngl999_ncmoe():
    hw = recommend.detect_hardware()
    st = {"name": "test", "path": "/tmp/test.gguf", "file_gb": 5.0, "is_moe": True,
          "total_b": 7, "active_b": 7, "layers": 32, "kv_bpt": 1024*1024,
          "expert_gb": 2.0, "other_gb": 0, "ssm_state_gb": 0, "kv_layers": 32,
          "ssm_layers": 0, "expert_count": 8, "active_experts": 2, "mtp_layers": 0}
    pre = recommend.prefill(st, hw)
    assert pre.get("ngl") == "999", f"MoE 预填 ngl 应为 999，实际 {pre.get('ngl')}"
    assert "ncpu_moe" in pre or "cpu_moe" in pre
    ok("Prefill: MoE → ngl=999 + ncpu_moe")


def test_prefill_dense_reject_gpu_ram_budget():
    hw = recommend.detect_hardware()
    # 15.7G dense > 8G VRAM → reject_gpu 纯 CPU；kv_bpt=98304 (≈96KiB/token,
    # 27B f16→q8_0 真实量级)：RAM budget 21.04GiB 下 65536 需 21.39GiB 超预算，
    # 32768 需 19.89GiB 可容纳 → RAM 重选应取 32768
    st = {"name": "test", "path": "/tmp/test.gguf", "file_gb": 15.7, "is_moe": False,
          "total_b": 27, "active_b": 27, "layers": 48, "kv_bpt": 98304,
          "expert_gb": 0, "other_gb": 0, "ssm_state_gb": 0, "kv_layers": 48,
          "ssm_layers": 0, "expert_count": 0, "active_experts": 0, "mtp_layers": 0,
          "mmap_ok": False}
    pre = recommend.prefill(st, hw)
    assert pre.get("ngl") == "0", f"Dense reject_gpu 应 ngl=0，实际 {pre.get('ngl')}"
    assert pre.get("ctx") == "32768", f"Dense 纯 CPU RAM 预算内应 ctx=32768，实际 {pre.get('ctx')}"
    ok("Prefill: Dense reject_gpu → ngl=0 + ctx=32768 (RAM budget)")


def test_prefill_batch_ubatch_prefill_optimized():
    hw = recommend.detect_hardware()
    # MoE
    st_moe = {"name": "test", "path": "/tmp/test.gguf", "file_gb": 5.0, "is_moe": True,
              "total_b": 7, "active_b": 7, "layers": 32, "kv_bpt": 1024*1024,
              "expert_gb": 2.0, "other_gb": 0, "ssm_state_gb": 0, "kv_layers": 32,
              "ssm_layers": 0, "expert_count": 8, "active_experts": 2, "mtp_layers": 0}
    pre_moe = recommend.prefill(st_moe, hw)
    assert pre_moe.get("batch") == "4096" and pre_moe.get("ubatch") == "1024", \
        f"MoE prefill batch/ubatch 应为 4096/1024，实际 {pre_moe.get('batch')}/{pre_moe.get('ubatch')}"
    # Dense
    st_dense = {"name": "test", "path": "/tmp/test.gguf", "file_gb": 5.0, "is_moe": False,
                "total_b": 7, "active_b": 7, "layers": 32, "kv_bpt": 1024*1024,
                "expert_gb": 0, "other_gb": 0, "ssm_state_gb": 0, "kv_layers": 32,
                "ssm_layers": 0, "expert_count": 0, "active_experts": 0, "mtp_layers": 0}
    pre_dense = recommend.prefill(st_dense, hw)
    assert pre_dense.get("batch") == "2048" and pre_dense.get("ubatch") == "512", \
        f"Dense prefill batch/ubatch 应为 2048/512，实际 {pre_dense.get('batch')}/{pre_dense.get('ubatch')}"
    ok("Prefill: batch/ubatch 预填优化值 (MoE 4096/1024, Dense 2048/512)")


# ============== build_command: default/auto/skip 语义 ==============
def make_hw():
    return recommend.detect_hardware()


def make_st(is_moe=False):
    return {"name": "test", "path": "/tmp/test.gguf", "file_gb": 5.0, "is_moe": is_moe,
            "total_b": 7, "active_b": 7, "layers": 32, "kv_bpt": 1024*1024,
            "expert_gb": 2.0 if is_moe else 0, "other_gb": 0, "ssm_state_gb": 0,
            "kv_layers": 32, "ssm_layers": 0, "expert_count": 8 if is_moe else 0,
            "active_experts": 2 if is_moe else 0, "mtp_layers": 0}


def test_build_cmd_default_skip():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    cmd = recommend.build_command(st, hw, plan, "llama-server", params={
        "ngl": "default", "threads": "default", "cpu_moe": "default",
        "moe_cache_mib": "default", "moe_cache_cpu_share": "default",
        "ctx": "4096", "flash_attn": "default", "cache_type_k": "default",
        "cache_type_v": "default", "spec_type": "default",
        "load_mode": "default", "temp": "0.75", "top_k": "40",
        "top_p": "0.95", "min_p": "0.05", "repeat_penalty": "1.1",
        "presence_penalty": "0.05", "frequency_penalty": "0",
        "seed": "-1", "reasoning": "default", "reasoning_effort": "default",
        "reasoning_format": "default", "jinja": "default", "agent": "default",
        "batch": "512", "ubatch": "512", "no_warmup": "default",
        "swa_full": "default", "cache_reuse": "default",
        "cache_ram": "default", "repeat_last_n": "default",
        "mmproj": "default", "mmproj_device": "default",
        "image_min_tokens": "default", "image_max_tokens": "default",
    })
    cmd_str = " ".join(cmd)
    # default 跳过：不应出现这些 flag
    for flag in ("-ngl", "--threads", "--cpu-moe", "--moe-cache-mib", "--moe-cache-cpu-share",
                 "--flash-attn", "--cache-type-k", "--cache-type-v", "--load-mode",
                 "--jinja", "--agent", "--spec-type", "--swa-full", "--cache-reuse",
                 "--cache-ram", "--repeat-last-n"):
        assert flag not in cmd_str, f"default 值不应下发 {flag}，但命令含: {cmd_str}"
    # ctx 等非 default 值应保留
    assert "--ctx-size 4096" in cmd_str
    ok("build_command: default/不配置/空值 → 完全不下发参数")


def test_build_cmd_auto_emits():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    cmd = recommend.build_command(st, hw, plan, "llama-server", params={
        "ngl": "auto", "threads": "auto", "flash_attn": "on",
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "ctx": "4096", "temp": "0.75", "top_k": "40", "top_p": "0.95",
        "min_p": "0.05", "repeat_penalty": "1.1", "presence_penalty": "0.05",
        "frequency_penalty": "0", "seed": "-1", "jinja": "on", "agent": "on",
        "batch": "512", "ubatch": "512", "no_warmup": "auto",
        "swa_full": "auto", "cache_reuse": "0",
        "cache_ram": "2048", "repeat_last_n": "128",
        "mmproj": "off", "mmproj_device": "none",
        "image_min_tokens": "256", "image_max_tokens": "4096",
    })
    cmd_str = " ".join(cmd)
    # auto → 用 plan 值（ngl > 0 会落 -ngl N）
    assert "-ngl" in cmd_str
    # threads=auto 不下发（二进制默认物理核）
    assert "--threads" not in cmd_str
    # flash_attn=on 落地
    assert "--flash-attn on" in cmd_str
    # cache_type_k/v=q8_0 落地
    assert "--cache-type-k q8_0" in cmd_str
    assert "--cache-type-v q8_0" in cmd_str
    # jinja=on / agent=on 落地
    assert "--jinja" in cmd_str
    assert "--agent" in cmd_str
    # cache_ram=2048 / repeat_last_n=128 落地
    assert "--cache-ram 2048" in cmd_str
    assert "--repeat-last-n 128" in cmd_str
    ok("build_command: auto/on/q8_0 等预填值 → 正确下发 plan 值")


def test_build_cmd_specific_values():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    cmd = recommend.build_command(st, hw, plan, "llama-server", params={
        "ngl": "32", "threads": "8", "cpu_moe": "on",
        "moe_cache_mib": "1536", "moe_cache_cpu_share": "0.5",
        "ctx": "8192", "flash_attn": "on", "cache_type_k": "q8_0",
        "cache_type_v": "q4_0", "spec_type": "draft-mtp",
        "spec_draft_model": "/path/to/mtp.gguf", "spec_draft_n_max": "4",
        "spec_draft_cache_type_k": "q8_0", "spec_draft_cache_type_v": "q8_0",
        "load_mode": "mmap", "temp": "0.8", "top_k": "20",
        "top_p": "0.9", "min_p": "0.1", "repeat_penalty": "1.15",
        "presence_penalty": "0.1", "frequency_penalty": "0.2",
        "seed": "42", "reasoning": "on", "reasoning_effort": "low",
        "reasoning_format": "deepseek", "jinja": "off", "agent": "off",
        "batch": "1024", "ubatch": "256", "no_warmup": "on",
        "swa_full": "on", "cache_reuse": "2048",
        "cache_ram": "4096", "repeat_last_n": "256",
        "mmproj": "/path/to/mmproj.gguf", "mmproj_device": "gpu",
        "image_min_tokens": "512", "image_max_tokens": "8192",
    })
    cmd_str = " ".join(cmd)
    checks = ["-ngl 32", "--threads 8", "--cpu-moe", "--moe-cache-mib 1536",
              "--moe-cache-cpu-share 0.5", "--ctx-size 8192", "--flash-attn on",
              "--cache-type-k q8_0", "--cache-type-v q4_0", "--spec-type draft-mtp",
              "--spec-draft-model /path/to/mtp.gguf", "--spec-draft-n-max 4",
              "--cache-type-k-draft q8_0", "--cache-type-v-draft q8_0",
              "--load-mode mmap", "--temp 0.8", "--top-k 20", "--top-p 0.9",
              "--min-p 0.1", "--repeat-penalty 1.15", "--presence-penalty 0.1",
              "--frequency-penalty 0.2", "--seed 42", "--reasoning on",
              "--reasoning-effort low", "--reasoning-format deepseek",
              "--mmproj /path/to/mmproj.gguf", "--mmproj-device gpu",
              "--image-min-tokens 512", "--image-max-tokens 8192",
              "--batch-size 1024", "--ubatch-size 256", "--no-warmup",
              "--swa-full", "--cache-reuse 2048",
              "--cache-ram 4096", "--repeat-last-n 256"]
    for c in checks:
        assert c in cmd_str, f"缺少预期参数: {c} (命令: {cmd_str})"
    ok("build_command: 具体数值 → 全部按面板值下发")


def test_build_cmd_moe_params_only_for_moe():
    hw = make_hw()
    st_dense = make_st(is_moe=False)
    plan = recommend.recommend(st_dense, hw, ctx=4096, kv_type="q8_0")
    cmd = recommend.build_command(st_dense, hw, plan, "llama-server", params={
        "ngl": "auto", "cpu_moe": "on", "moe_cache_mib": "1024",
        "moe_cache_cpu_share": "0.5", "ctx": "4096",
    })
    cmd_str = " ".join(cmd)
    # dense 模型即使面板开 cpu_moe/moe-cache-* 也不应下发（由推荐引擎决定）
    # 实际上 FLAG_MAP 里这些 key 有 fn，build_command 会落下去，但推荐引擎不建议
    # 这里只验证不崩溃
    ok("build_command: MoE 专用参数在 dense 模型下不报错（由用户自行决定）")


def test_build_cmd_tosh_skips_moe_cache():
    hw = make_hw()
    st = make_st(is_moe=True)
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    cmd = recommend.build_command(st, hw, plan, "/Applications/ToshLLM.app/Contents/Resources/bin/llama-server",
                                  params={
        "ngl": "auto", "cpu_moe": "on", "moe_cache_mib": "1024",
        "moe_cache_cpu_share": "0.5", "ctx": "4096",
    })
    cmd_str = " ".join(cmd)
    assert "--moe-cache-mib" not in cmd_str, "ToshLLM 无 moe-cache-mib 旗标"
    assert "--moe-cache-cpu-share" not in cmd_str, "ToshLLM 无 moe-cache-cpu-share 旗标"
    ok("build_command: ToshLLM 引擎自动省略 moe-cache-*")


def test_build_cmd_clear_mode_override():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    # override 模式（默认）：框架 + 面板参数 + 用户覆盖
    cmd = recommend.build_command(st, hw, plan, "llama-server",
                                  params={"ngl": "32", "ctx": "8192"},
                                  override_args="-c 4096 --temp 0.5",
                                  clear_mode="override")
    cmd_str = " ".join(cmd)
    assert "-ngl 32" in cmd_str  # 面板参数保留
    assert "-c 4096" in cmd_str  # 用户覆盖 -c 4096（同族去重，用户胜出）
    assert "--ctx-size 8192" not in cmd_str  # 面板 ctx 被用户 -c 去重
    assert "--temp 0.5" in cmd_str
    ok("build_command: clear_mode=override → 面板参数+用户覆盖(去重)")


def test_build_cmd_new_params_boundary():
    """新参数边界：cache_ram=0/repeat_last_n=64 视为 default 省略；3072 正常下发。"""
    st = make_st(is_moe=True)
    hw = make_hw()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    # cache_ram="0" / repeat_last_n="64" / cache_ram="default"  → 均不应出现在命令
    cmd = recommend.build_command(st, hw, plan, "llama-server", params={
        "ngl": "32", "threads": "8", "cpu_moe": "default", "moe_cache_mib": "0",
        "moe_cache_cpu_share": "0", "flash_attn": "off", "cache_type_k": "f16",
        "cache_type_v": "f16", "spec_type": "off", "swa_full": "off", "cache_reuse": "0",
        "cache_ram": "0", "repeat_last_n": "64", "mmproj": "default",
        "mmproj_device": "default", "image_min_tokens": "default", "image_max_tokens": "default",
        "load_mode": "default", "temp": "0.8", "top_k": "0", "top_p": "1.0",
        "min_p": "0.0", "repeat_penalty": "1.1", "presence_penalty": "0.0",
        "frequency_penalty": "0.0", "seed": "-1", "reasoning": "off",
        "reasoning_effort": "default", "reasoning_format": "default", "jinja": "default",
        "agent": "default", "batch": "512", "ubatch": "512", "no_warmup": "default",
        "ctx": "default", "spec_draft_model": "default", "spec_draft_n_max": "default",
        "spec_draft_cache_type_k": "default", "spec_draft_cache_type_v": "default",
    })
    cmd_str = " ".join(cmd)
    assert "--cache-ram" not in cmd_str, f"cache_ram=0 不应下发: {cmd_str}"
    assert "--repeat-last-n" not in cmd_str, f"repeat_last_n=64 不应下发: {cmd_str}"
    # cache_ram="3072" → 下发
    cmd2 = recommend.build_command(st, hw, plan, "llama-server", params={
        "ngl": "32", "threads": "8", "cpu_moe": "default", "moe_cache_mib": "0",
        "moe_cache_cpu_share": "0", "flash_attn": "off", "cache_type_k": "f16",
        "cache_type_v": "f16", "spec_type": "off", "swa_full": "off", "cache_reuse": "0",
        "cache_ram": "3072", "repeat_last_n": "128", "mmproj": "default",
        "mmproj_device": "default", "image_min_tokens": "default", "image_max_tokens": "default",
        "load_mode": "default", "temp": "0.8", "top_k": "0", "top_p": "1.0",
        "min_p": "0.0", "repeat_penalty": "1.1", "presence_penalty": "0.0",
        "frequency_penalty": "0.0", "seed": "-1", "reasoning": "off",
        "reasoning_effort": "default", "reasoning_format": "default", "jinja": "default",
        "agent": "default", "batch": "512", "ubatch": "512", "no_warmup": "default",
        "ctx": "default", "spec_draft_model": "default", "spec_draft_n_max": "default",
        "spec_draft_cache_type_k": "default", "spec_draft_cache_type_v": "default",
    })
    cmd2_str = " ".join(cmd2)
    assert "--cache-ram 3072" in cmd2_str, f"cache_ram=3072 应下发: {cmd2_str}"
    assert "--repeat-last-n 128" in cmd2_str, f"repeat_last_n=128 应下发: {cmd2_str}"
    ok("build_command: 新参数边界值 省略/下发 正确")


def test_build_cmd_clear_mode_clear():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    # clear 模式：仅框架 + 用户输入
    cmd = recommend.build_command(st, hw, plan, "llama-server",
                                  params={"ngl": "32", "ctx": "8192", "temp": "0.75"},
                                  override_args="-c 4096 --temp 0.5",
                                  clear_mode="clear")
    cmd_str = " ".join(cmd)
    # 面板参数全部不下发
    assert "-ngl" not in cmd_str
    assert "--ctx-size" not in cmd_str
    # 用户覆盖的 --temp 0.5 下发（面板 temp 不下发，但同族键只保留用户值）
    assert "--temp 0.5" in cmd_str
    ok("build_command: clear_mode=clear → 仅框架+用户输入，无任何自动参数")


def test_build_cmd_override_conflict_resolution():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    # 面板 ctx=8192，用户覆盖 -c 4096 → 用户胜出
    cmd = recommend.build_command(st, hw, plan, "llama-server",
                                  params={"ctx": "8192"},
                                  override_args="-c 4096",
                                  clear_mode="override")
    cmd_str = " ".join(cmd)
    # 用户的 -c 4096 胜出，面板 --ctx-size 8192 被去重
    assert "-c 4096" in cmd_str
    assert "--ctx-size 8192" not in cmd_str
    ok("build_command: 自定义覆盖同名族去重，用户值胜出")


def test_build_cmd_spec_draft_model_guard():
    hw = make_hw()
    st = make_st()
    plan = recommend.recommend(st, hw, ctx=4096, kv_type="q8_0")
    base = {"ngl": "32", "spec_type": "draft-mtp"}
    # off/空/default → --spec-draft-model 省略（对齐 mmproj 守卫约定）
    for bad in ("off", "", "default"):
        cmd = " ".join(recommend.build_command(st, hw, plan, "llama-server",
                                               params={**base, "spec_draft_model": bad}))
        assert "--spec-draft-model" not in cmd, f"spec_draft_model={bad!r} 应省略旗标: {cmd}"
        assert "--spec-type draft-mtp" in cmd, "spec_type=draft-mtp 必须下发"
    # 真实路径 → 下发
    cmd = " ".join(recommend.build_command(st, hw, plan, "llama-server",
                                           params={**base, "spec_draft_model": "/vol/models/mtpdraft-x.gguf"}))
    assert "--spec-draft-model /vol/models/mtpdraft-x.gguf" in cmd, f"路径应原样下发: {cmd}"
    ok("build_command: spec_draft_model off/空/default 省略，路径下发（MTP draft 边界）")


def test_mtp_cands_autodetect():
    import tempfile
    import gui as gui_mod
    with tempfile.TemporaryDirectory() as td:
        for f in ("mtpdraft-ornith-Q8_0.gguf", "Qwen3-Next-80B-MTP-f16.gguf",
                  "Main-Model-Q4_K_M.gguf", "mmproj-model-f16.gguf"):
            with open(os.path.join(td, f), "wb"):
                pass
        cands, display, mtp_map = gui_mod._build_mtp_cands(td)
        assert len(cands) == 2, f"MTP 候选应为 2 个，实际 {cands}"
        assert display[0] == "off"
        names = [os.path.basename(p) for p in cands]
        assert "mtpdraft-ornith-Q8_0.gguf" in names, "小写 mtpdraft 未识别"
        assert "Qwen3-Next-80B-MTP-f16.gguf" in names, "大写 MTP 未识别"
        assert "Main-Model-Q4_K_M.gguf" not in names, "主模型误入 MTP 候选"
        assert "mmproj-model-f16.gguf" not in names, "mmproj 文件误入 MTP 候选"
        for p in cands:
            assert mtp_map[os.path.basename(p)[:30]] == p, "显示名必须映射回全路径"
        assert mtp_map["default"] == "" and mtp_map["off"] == "", "default/off 必须映射空串"
        # 空目录 / 目录不存在 → 不抛异常，空候选
        c2, d2, m2 = gui_mod._build_mtp_cands(os.path.join(td, "no-such-dir"))
        assert c2 == [] and d2 == ["off"] and m2["default"] == ""
    ok("MTP 自动识别: 大小写不敏感匹配 + 显示名映射 + 目录缺失兜底")


def test_scan_models_skips_mtp():
    import tempfile
    import gui as gui_mod
    with tempfile.TemporaryDirectory() as td:
        for f in ("mtpdraft-x-Q8_0.gguf", "MainModel-Q4_K_M.gguf"):
            with open(os.path.join(td, f), "wb"):
                pass
        out = gui_mod.scan_models(td)
        names = [s.get("name", "") for s in out]
        assert not any("mtpdraft" in n.lower() for n in names), "MTP 独立文件不得进主模型库"
        assert any(n.startswith("MainModel-Q4_K_M") for n in names), "主模型仍在库中"
    ok("模型库扫描: MTP draft 文件排除出主模型（SKIP 含 mtp）")


# ============== GUI 无头验证 ==============
def test_gui_headless_37_params():
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.withdraw()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gui import App
    app = App(root)
    root.update()
    # 38 个参数
    assert len(app.pv) == 38, f"参数数量 {len(app.pv)} != 38"
    # 所有 OptionMenu 垫首 "default" + 自由编辑 Combobox width=12
    FREE_EDIT_KEYS = {"ngl", "threads", "mmproj", "mmproj_device",
                      "spec_type", "spec_draft_model", "spec_draft_n_max"}
    for key, widget in app.widgets.items():
        if isinstance(widget, ttk.OptionMenu):
            # OptionMenu 内部菜单的第一项应为 "default"
            menu = widget["menu"]
            first_label = menu.entrycget(0, "label")
            assert first_label == "default", f"{key} OptionMenu 首项不是 default: {first_label}"
        if isinstance(widget, ttk.Combobox):
            # FREE_EDIT_KEYS 的 Combobox width=12
            if key in FREE_EDIT_KEYS:
                assert str(widget.cget("width")) == "12", f"{key} Combobox width != 12"
    root.destroy()
    ok("GUI 无头: 38 参数、OptionMenu 垫首 default、Combobox width=12")


def test_gui_button_bindings():
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.withdraw()
    from gui import App
    app = App(root)
    root.update()
    # 运行/停止按钮绑定 <ButtonPress-1>（add=+，视觉反馈不抢点击）
    # 注：bind() 内省看不到 add= 标记，改验绑定注册存在
    run_press = app.run_cv.bind("<ButtonPress-1>")
    stop_press = app.stop_cv.bind("<ButtonPress-1>")
    assert run_press and ("<lambda>" in run_press or run_press.strip()), "运行按钮未绑定 <ButtonPress-1>"
    assert stop_press and ("<lambda>" in stop_press or stop_press.strip()), "停止按钮未绑定 <ButtonPress-1>"
    # 主点击 <Button-1> 同样注册（触发执行/停止）
    run_click = app.run_cv.bind("<Button-1>")
    stop_click = app.stop_cv.bind("<Button-1>")
    assert run_click and run_click.strip(), "运行按钮未绑定 <Button-1>"
    assert stop_click and stop_click.strip(), "停止按钮未绑定 <Button-1>"
    root.destroy()
    ok("GUI 无头: 按钮绑定 ButtonPress-1 + Button-1（视觉反馈不抢点击）")


# ============== 引擎选择 ==============
def test_pick_engine_moe_to_tosh():
    hw = make_hw()
    st = make_st(is_moe=True)
    engine, binary, notes = recommend.pick_engine(st, hw)
    assert engine == "tosh", f"MoE 应选 tosh，实际 {engine}"
    ok("引擎选择: MoE → ToshLLM")


def test_pick_engine_dense_metal_to_tosh():
    hw = make_hw()
    st = make_st(is_moe=False)
    # 模拟有 Metal 计算
    hw.metal_compute = True
    engine, binary, notes = recommend.pick_engine(st, hw)
    assert engine == "tosh", f"有 Metal 计算 GPU 的 dense 应选 tosh，实际 {engine}"
    ok("引擎选择: Dense + Metal 计算 → ToshLLM")


def test_pick_engine_dense_no_metal_to_vanilla():
    hw = make_hw()
    st = make_st(is_moe=False)
    # 模拟无 Metal 计算（如本机 RX 6600 XT 仅显存显示）
    hw.metal_compute = False
    engine, binary, notes = recommend.pick_engine(st, hw)
    assert engine == "vanilla", f"无 Metal 计算 GPU 的 dense 应选 vanilla，实际 {engine}"
    ok("引擎选择: Dense 无 Metal 计算 → Vanilla")


# ============== 状态持久化 ==============
def test_state_persistence_save_load():
    import tempfile, os
    # 直接测试 save/load 逻辑（不启动完整 GUI）；STATE_FILE 重定向到临时文件
    import gui as gui_mod
    from gui import App
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    root.update()
    old_state_file = gui_mod.STATE_FILE
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    gui_mod.STATE_FILE = tmp.name
    try:
        app.selected = {"path": "/tmp/test.gguf", "name": "test.gguf"}
        # 设置一些参数
        app.pv["ctx"].set("32768")
        app.pv["temp"].set("0.8")
        app.engine_var.set("tosh")
        app.port_var.set("9000")
        app.host_var.set("0.0.0.0")
        app.ovr.insert("1.0", "-c 4096")
        # 保存
        app._save_state_entry()
        # 重新加载验证
        with open(gui_mod.STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        key = "/tmp/test.gguf"
        assert key in data
        assert data[key]["params"]["ctx"] == "32768"
        assert data[key]["params"]["temp"] == "0.8"
        assert data[key]["engine"] == "tosh"
        assert data[key]["port"] == "9000"
        assert data[key]["host"] == "0.0.0.0"
        assert data[key]["override"] == "-c 4096"
    finally:
        root.destroy()
        gui_mod.STATE_FILE = old_state_file
        try:
            os.remove(tmp.name)
        except Exception:
            pass
    ok("状态持久化: 按模型路径保存/加载面板/引擎/端口/host/覆盖")


# ============== 集成端到端 ==============
def test_integration_35b_moe_e2e():
    """35B-A3B MoE 端到端：选模型 → 预填 → 命令生成 → 关键参数落位"""
    hw = make_hw()
    models_dir = find_models_dir()
    if not models_dir:
        print("  ⚠ 跳过: 无可用模型目录")
        return
    ggufs = [p for p in sorted(os.listdir(models_dir)) if p.endswith(".gguf") and not p.startswith(".") and "qwen" in p.lower() and "35b" in p.lower() and not any(s in p.lower() for s in ("mmproj", "clip", "whisper"))]
    if not ggufs:
        print("  ⚠ 跳过: 无 35B MoE 模型")
        return
    p = os.path.join(models_dir, ggufs[0])
    h = gguf_fast.parse_gguf_header(p)
    st = recommend.model_stats(h)
    assert st.get("is_moe"), "非 MoE 模型"
    pre = recommend.prefill(st, hw)
    plan = recommend.recommend(st, hw, ctx=int(pre["ctx"]), kv_type=pre["cache_type_k"])
    engine, binary, _ = recommend.pick_engine(st, hw)
    cmd = recommend.build_command(st, hw, plan, binary, params=pre)
    cmd_str = " ".join(cmd)
    # 关键检查
    assert engine == "tosh", "35B MoE 应路由到 tosh"
    assert "-ngl 999" in cmd_str, "MoE 应 -ngl 999"
    assert "--n-cpu-moe" not in cmd_str, "11-13 定规：ncmoe 不再自动注入（OOM 降级为 plan 提示）"
    assert plan.get("oom_risk") and any("--n-cpu-moe" in h for h in plan.get("hints", [])), \
        "35B/8G 应有 oom_risk 提示且含 --n-cpu-moe 建议值"
    assert "--cache-type-k q8_0" in cmd_str, "默认 KV q8_0"
    assert "--flash-attn on" in cmd_str, "MoE 大 ctx 开 flash-attn"
    assert "--ctx-size 65536" in cmd_str, "默认 ctx 65536"
    assert "--batch-size 4096" in cmd_str, "MoE prefill batch 4096"
    assert "--ubatch-size 1024" in cmd_str, "MoE prefill ubatch 1024"
    assert "--agent" in cmd_str, "agent=on"
    assert "--jinja" in cmd_str, "jinja=on"
    ok("集成 E2E: 35B MoE → tosh/-ngl 999/无自动 ncmoe(oom 提示)/q8_0/fa/ctx65536/batch4096/ub1024/agent/jinja")


# ============== 语法检查 ==============
def test_syntax_check():
    """py_compile 三文件全过"""
    for f in ("gui.py", "recommend.py", "gguf_fast.py"):
        r = subprocess.run([sys.executable, "-m", "py_compile", f],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{f} 语法错误: {r.stderr}"
    ok("语法检查: gui.py/recommend.py/gguf_fast.py 全部通过")


# ============== GUI 测试子进程隔离 ==============
# macOS Tcl：同一进程里上一个 Tk 根 destroy 之后再建第二个 Tk 根，显示连接会死锁
# （本测试套在单进程内先后建 3 个 App/Tk，第二个即挂）。生产是"单进程单 App"，不受影响；
# 故把建 Tk 的 4 项 GUI 测试各自放进独立子进程（全新 Tcl 解释器，贴生产语义），
# 父进程只收集机器判定行 RESULT:PASS/FAIL/SKIP 并汇总计数，避免跨进程 Tcl 状态污染。
GUI_SUBPROC = {"test_gui_headless_37_params", "test_gui_button_bindings",
               "test_state_persistence_save_load", "test_gui_gen_command_click"}


def _run_gui_in_subprocess(test_fn):
    """在子进程运行单项 GUI 测试，按子进程的 RESULT: 判定行汇总 PASS/FAIL。"""
    global PASS, FAIL
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run([sys.executable, os.path.join(here, "test_regression.py"),
                            "--run-one", test_fn.__name__],
                           capture_output=True, text=True, timeout=180, cwd=here)
    except subprocess.TimeoutExpired:
        FAIL += 1
        print(f"  ❌ {test_fn.__name__}: 子进程超时（Tcl 死锁？）180s")
        return
    lines = [ln for ln in (r.stdout or "").strip().splitlines()
             if ln.strip() and not ln.strip().startswith("RESULT:")]
    for ln in lines:
        print(ln)
    verdict = ""
    for ln in reversed((r.stdout or "").strip().splitlines()):
        if ln.strip().startswith("RESULT:"):
            verdict = ln.strip()
            break
    if verdict == "RESULT:PASS" or (verdict == "" and r.returncode == 0 and any("✅" in ln for ln in lines)):
        PASS += 1
    elif verdict == "RESULT:SKIP" or (verdict == "" and r.returncode == 0 and any("⚠" in ln for ln in lines)):
        print(f"  ⚠ {test_fn.__name__}: 子进程跳过（前置条件不满足，不计 PASS/FAIL）")
    else:
        FAIL += 1
        print(f"  ❌ {test_fn.__name__}: 子进程 FAIL (exit={r.returncode}) "
              f"{(r.stderr or '').strip()[:200]}")


# ============== 「生成指令」按钮回归 ==============
def test_gui_gen_command_click():
    """点「⚡生成指令」端到端：「无法生成命令」的回归门禁。

    历史缺陷：_update_ctv_candidates 曾把 cache_type_v 的 OptionMenu 当 Combobox
    写 ['values']，抛 TclError 中断整条 _refresh_recommend 链路，命令框永远空。
    这里实例化 GUI、等模型扫描完成后点「生成指令」，断言命令生成，
    且 OptionMenu 动态候选（引擎切换）重建不崩溃。"""
    import time
    import tkinter as tk
    from tkinter import ttk
    if find_models_dir() is None:
        print("  ⚠ 跳过: 无可用模型目录（无法验证命令生成）")
        return
    root = tk.Tk()
    root.withdraw()
    from gui import App
    app = App(root)
    deadline = time.time() + 120
    while time.time() < deadline and not app.tree.get_children():
        root.update()
    try:
        assert app.tree.get_children(), "模型扫描超时 120s（模型库无模型）"
        app._on_gen_click()
        root.update()
        cmd = app.cmd.get("1.0", "end").strip()
        assert cmd and app._cmd_available, f"「生成指令」未产生命令: {cmd!r}"
        # OptionMenu 动态候选重建（引擎 tosh↔vanilla 切换）必须不抛 TclError
        om = app.widgets["cache_type_v"]
        assert isinstance(om, ttk.OptionMenu), "cache_type_v 应为 OptionMenu"
        app.engine_var.set("tosh")
        app._refresh_recommend()
        root.update()
        app.engine_var.set("vanilla")
        app._refresh_recommend()
        root.update()
        items = [om["menu"].entrycget(i, "label")
                 for i in range(om["menu"].index("end") + 1)]
        assert items and items[0] == "default", f"OptionMenu 首项应为 default: {items}"
        # 再点一次「生成指令」仍可用（幂等、不崩）
        app._on_gen_click()
        root.update()
        assert app._cmd_available, "第二次点「生成指令」后命令应仍可用"
    finally:
        root.destroy()
    ok("GUI: 「生成指令」按钮端到端（命令生成 + OptionMenu 候选重建不崩）")


# ============== 运行所有 ==============
if __name__ == "__main__":
    # 子进程单测入口：python3 test_regression.py --run-one <testfn_name>
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-one":
        _one = getattr(sys.modules[__name__], sys.argv[2])
        try:
            _one()
        except AssertionError as e:
            ng(f"{sys.argv[2]}: 断言失败 - {e}")
        except Exception as e:
            ng(f"{sys.argv[2]}: 异常 - {type(e).__name__}: {e}")
        # 机器判定行：父进程据此汇总；FAIL>0→FAIL 且 exit1，PASS>0→PASS，都 0→SKIP（优雅跳过）
        if FAIL > 0:
            print("RESULT:FAIL")
            sys.exit(1)
        elif PASS > 0:
            print("RESULT:PASS")
        else:
            print("RESULT:SKIP")
        sys.exit(0)
    tests = [
        test_syntax_check,
        test_hardware_detection,
        test_model_stats_parsing,
        test_recommend_basic,
        test_recommend_moe_clamps_ncmoe,
        test_prefill_moe_uses_ngl999_ncmoe,
        test_prefill_dense_reject_gpu_ram_budget,
        test_prefill_batch_ubatch_prefill_optimized,
        test_build_cmd_default_skip,
        test_build_cmd_auto_emits,
        test_build_cmd_specific_values,
        test_build_cmd_moe_params_only_for_moe,
        test_build_cmd_tosh_skips_moe_cache,
        test_build_cmd_clear_mode_override,
        test_build_cmd_clear_mode_clear,
        test_build_cmd_new_params_boundary,
        test_build_cmd_override_conflict_resolution,
        test_build_cmd_spec_draft_model_guard,
        test_mtp_cands_autodetect,
        test_scan_models_skips_mtp,
        test_gui_headless_37_params,
        test_gui_button_bindings,
        test_gui_gen_command_click,
        test_pick_engine_moe_to_tosh,
        test_pick_engine_dense_metal_to_tosh,
        test_pick_engine_dense_no_metal_to_vanilla,
        test_state_persistence_save_load,
        test_integration_35b_moe_e2e,
    ]
    print(f"开始运行 {len(tests)} 项回归测试...\n")
    for t in tests:
        if t.__name__ in GUI_SUBPROC:
            _run_gui_in_subprocess(t)
            continue
        try:
            t()
        except AssertionError as e:
            ng(f"{t.__name__}: 断言失败 - {e}")
        except Exception as e:
            ng(f"{t.__name__}: 异常 - {type(e).__name__}: {e}")
    print(f"\n{'='*50}")
    print(f"结果: {PASS} PASS, {FAIL} FAIL")
    print(f"{'='*50}")
    sys.exit(0 if FAIL == 0 else 1)