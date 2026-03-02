#!/usr/bin/env python3
"""Lab 5: Quantization Impact

Goals:
  - Compare FP16 vs INT8 (AWQ) quantized inference
  - Measure memory savings, throughput changes, and quality differences
  - Send identical prompts to both and compare outputs side by side

Usage:
  # This lab manages its own vLLM servers.
  # Make sure no other vLLM server is running on port 8000.
  python -m benchmarks.lab5_quantization

  # Or with pre-started server:
  python -m benchmarks.lab5_quantization --server-running --model-type fp16
  python -m benchmarks.lab5_quantization --server-running --model-type int8
"""

import argparse
import subprocess
import sys
import time

from benchmarks.utils import (
    configure_vllm_env,
    get_gpu_memory_mb,
    gpu_info_snapshot,
    load_config,
    make_prompt,
    plot_comparison_bars,
    print_table,
    save_results,
    stream_chat_completion,
    wait_for_server,
)


# Quality-comparison prompts — short, evaluable outputs
QUALITY_PROMPTS = [
    "What are three benefits of regular exercise? Be concise.",
    "Explain photosynthesis to a 10-year-old in two sentences.",
    "Write a haiku about machine learning.",
    "What is the capital of France and why is it culturally important? One paragraph.",
    "Suggest a healthy breakfast recipe with eggs. Keep it under 50 words.",
]


def start_vllm_server(model: str, port: int, quantization: str | None = None) -> subprocess.Popen:
    configure_vllm_env()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--enforce-eager",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    print(f"  Starting server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open("/tmp/vllm_stderr.log", "w"))


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  Server stopped.")


def benchmark_config(base_url: str, model: str, concurrency_prompt_tokens: int,
                     output_tokens: int, num_runs: int = 5) -> dict:
    """Run a performance benchmark: TTFT, throughput, memory."""
    prompt = make_prompt(concurrency_prompt_tokens)
    messages = [{"role": "user", "content": prompt}]

    ttfts, totals, tps_list = [], [], []
    for i in range(num_runs):
        m = stream_chat_completion(
            base_url=base_url, model=model,
            messages=messages, max_tokens=output_tokens,
        )
        ttfts.append(m.ttft_ms)
        totals.append(m.total_time_ms)
        tps_list.append(m.tokens_per_sec)

    return {
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2),
        "avg_total_ms": round(sum(totals) / len(totals), 2),
        "avg_tps": round(sum(tps_list) / len(tps_list), 2),
        "gpu_mem_mb": get_gpu_memory_mb(),
    }


def get_quality_outputs(base_url: str, model: str) -> list[str]:
    """Get model outputs for the quality comparison prompts."""
    outputs = []
    for prompt_text in QUALITY_PROMPTS:
        m = stream_chat_completion(
            base_url=base_url, model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=150,
            temperature=0.3,  # low temp for reproducibility
        )
        # We need the actual text — use non-streaming for this
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="not-needed")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=150,
            temperature=0.3,
        )
        outputs.append(resp.choices[0].message.content.strip())
    return outputs


def run_lab5():
    parser = argparse.ArgumentParser(description="Lab 5: Quantization Impact")
    parser.add_argument("--server-running", action="store_true")
    parser.add_argument("--model-type", choices=["fp16", "int8"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    base_url = cfg["server"]["base_url"]
    port = cfg["server"]["port"]
    lab_cfg = cfg["lab5"]

    fp16_model = cfg["default_model"]
    int8_model = lab_cfg["quantized_model"]
    prompt_tokens = lab_cfg["prompt_tokens"]
    output_tokens = lab_cfg["output_tokens"]

    print("=" * 60)
    print("  Lab 5: Quantization Impact (FP16 vs INT8)")
    print("=" * 60)

    gpu_info = gpu_info_snapshot()
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['total_mem_mb']:.0f} MiB total)")
    print(f"  FP16 model: {fp16_model}")
    print(f"  INT8 model: {int8_model}")

    configs_to_run = []
    if args.server_running:
        if args.model_type == "fp16":
            configs_to_run = [("FP16", fp16_model, None)]
        elif args.model_type == "int8":
            configs_to_run = [("INT8 (AWQ)", int8_model, "awq")]
        else:
            configs_to_run = [("FP16", fp16_model, None)]
    else:
        configs_to_run = [
            ("FP16", fp16_model, None),
            ("INT8 (AWQ)", int8_model, "awq"),
        ]

    all_perf = {}
    all_quality = {}

    for label, model, quant in configs_to_run:
        print(f"\n{'─' * 50}")
        print(f"  Testing: {label} — {model}")
        print(f"{'─' * 50}")

        proc = None
        if not args.server_running:
            proc = start_vllm_server(model, port, quant)

        if not wait_for_server(base_url, timeout=300):
            print(f"  ERROR: Server failed to start for {label}")
            if proc:
                stop_server(proc)
            continue

        # Warmup
        stream_chat_completion(
            base_url=base_url, model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )

        # Performance benchmark
        print("  Running performance benchmark...")
        perf = benchmark_config(base_url, model, prompt_tokens, output_tokens)
        all_perf[label] = perf
        print(f"    TTFT:     {perf['avg_ttft_ms']:.1f} ms")
        print(f"    TPS:      {perf['avg_tps']:.1f}")
        print(f"    GPU mem:  {perf['gpu_mem_mb']:.0f} MiB")

        # Quality comparison
        print("  Getting quality comparison outputs...")
        quality = get_quality_outputs(base_url, model)
        all_quality[label] = quality

        if proc:
            stop_server(proc)
            time.sleep(5)

    # ── Performance comparison ────────────────────────────────
    if len(all_perf) == 2:
        labels = list(all_perf.keys())
        p_a = all_perf[labels[0]]
        p_b = all_perf[labels[1]]

        print_table(
            headers=["Metric", labels[0], labels[1], "Difference"],
            rows=[
                ["TTFT (ms)", f"{p_a['avg_ttft_ms']:.1f}", f"{p_b['avg_ttft_ms']:.1f}",
                 f"{((p_b['avg_ttft_ms'] - p_a['avg_ttft_ms']) / p_a['avg_ttft_ms'] * 100):+.1f}%"],
                ["TPS", f"{p_a['avg_tps']:.1f}", f"{p_b['avg_tps']:.1f}",
                 f"{((p_b['avg_tps'] - p_a['avg_tps']) / p_a['avg_tps'] * 100):+.1f}%"],
                ["GPU Mem (MiB)", f"{p_a['gpu_mem_mb']:.0f}", f"{p_b['gpu_mem_mb']:.0f}",
                 f"{((p_b['gpu_mem_mb'] - p_a['gpu_mem_mb']) / p_a['gpu_mem_mb'] * 100):+.1f}%"],
            ],
            title="Lab 5: FP16 vs INT8 Performance",
        )

        mem_saved = p_a["gpu_mem_mb"] - p_b["gpu_mem_mb"]
        print("Key Observations:")
        print(f"  - Memory saved: {mem_saved:.0f} MiB "
              f"({mem_saved / p_a['gpu_mem_mb'] * 100:.0f}% reduction)")
        print(f"  - TTFT change: {p_a['avg_ttft_ms']:.1f}ms → {p_b['avg_ttft_ms']:.1f}ms")
        print(f"  - Throughput change: {p_a['avg_tps']:.1f} → {p_b['avg_tps']:.1f} tok/s")
        print(f"  - Quantization trades precision for efficiency — a real production decision")

        plot_comparison_bars(
            labels=["TTFT (ms)", "TPS", "GPU Mem (GiB)"],
            values_a=[p_a["avg_ttft_ms"], p_a["avg_tps"], p_a["gpu_mem_mb"] / 1024],
            values_b=[p_b["avg_ttft_ms"], p_b["avg_tps"], p_b["gpu_mem_mb"] / 1024],
            label_a=labels[0],
            label_b=labels[1],
            y_label="Value",
            title="Lab 5: FP16 vs INT8 Comparison",
            filename="lab5_quantization_comparison.png",
        )

    # ── Quality comparison ────────────────────────────────────
    if len(all_quality) == 2:
        labels = list(all_quality.keys())
        print(f"\n{'=' * 60}")
        print(f"  Quality Comparison: {labels[0]} vs {labels[1]}")
        print(f"{'=' * 60}")
        for i, prompt_text in enumerate(QUALITY_PROMPTS):
            print(f"\n  Prompt: \"{prompt_text}\"")
            print(f"  {labels[0]}: {all_quality[labels[0]][i][:200]}")
            print(f"  {labels[1]}: {all_quality[labels[1]][i][:200]}")
            print(f"  {'─' * 40}")

    # ── Save ──────────────────────────────────────────────────
    save_results(
        {
            "lab": "lab5_quantization",
            "gpu": gpu_info,
            "performance": all_perf,
            "quality_prompts": QUALITY_PROMPTS,
            "quality_outputs": {k: v for k, v in all_quality.items()},
        },
        "lab5_results.json",
    )
    print("\nLab 5 complete!")


if __name__ == "__main__":
    run_lab5()
