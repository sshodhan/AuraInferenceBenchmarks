#!/usr/bin/env python3
"""Lab 1: Deploy Your First Model with vLLM

Goals:
  - Install vLLM and download a model
  - Start an OpenAI-compatible API server
  - Send a test prompt and measure TTFT + total generation time
  - Observe GPU memory usage during model loading and inference

Usage:
  # Step 1: Start the vLLM server (in a separate terminal or background):
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --port 8000

  # Step 2: Run this benchmark:
  python -m benchmarks.lab1_deploy
"""

import sys
import time

from benchmarks.utils import (
    gpu_info_snapshot,
    load_config,
    make_prompt,
    print_table,
    save_results,
    stream_chat_completion,
    wait_for_server,
)


def run_lab1():
    cfg = load_config()
    base_url = cfg["server"]["base_url"]
    model = cfg["default_model"]

    print("=" * 60)
    print("  Lab 1: Deploy Your First Model with vLLM")
    print("=" * 60)

    # ── GPU baseline ──────────────────────────────────────────
    print("\n[1/4] GPU info before inference:")
    gpu_before = gpu_info_snapshot()
    for k, v in gpu_before.items():
        print(f"  {k}: {v}")

    # ── Wait for server ───────────────────────────────────────
    print("\n[2/4] Checking server availability...")
    if not wait_for_server(base_url):
        print(
            "\nServer not reachable. Start it with:\n"
            f"  python -m vllm.entrypoints.openai.api_server "
            f"--model {model} --port {cfg['server']['port']}"
        )
        sys.exit(1)

    # ── Short prompt test ─────────────────────────────────────
    print("\n[3/4] Sending a short test prompt (~50 tokens)...")
    short_prompt = "Explain what a GPU is in two sentences."
    messages = [{"role": "user", "content": short_prompt}]

    metrics_short = stream_chat_completion(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=100,
    )
    gpu_during = gpu_info_snapshot()

    print(f"  TTFT:             {metrics_short.ttft_ms:>8.1f} ms")
    print(f"  Total time:       {metrics_short.total_time_ms:>8.1f} ms")
    print(f"  Tokens generated: {metrics_short.tokens_generated:>8d}")
    print(f"  Tokens/sec:       {metrics_short.tokens_per_sec:>8.1f}")
    print(f"  GPU memory:       {gpu_during['used_mem_mb']:>8.0f} MiB")

    # ── Long prompt test ──────────────────────────────────────
    print("\n[4/4] Sending a longer prompt (~500 tokens)...")
    long_prompt = make_prompt(500)
    messages_long = [{"role": "user", "content": long_prompt}]

    metrics_long = stream_chat_completion(
        base_url=base_url,
        model=model,
        messages=messages_long,
        max_tokens=100,
    )
    gpu_after = gpu_info_snapshot()

    print(f"  TTFT:             {metrics_long.ttft_ms:>8.1f} ms")
    print(f"  Total time:       {metrics_long.total_time_ms:>8.1f} ms")
    print(f"  Tokens generated: {metrics_long.tokens_generated:>8d}")
    print(f"  Tokens/sec:       {metrics_long.tokens_per_sec:>8.1f}")
    print(f"  GPU memory:       {gpu_after['used_mem_mb']:>8.0f} MiB")

    # ── Summary table ─────────────────────────────────────────
    print_table(
        headers=["Metric", "Short Prompt (~50 tok)", "Long Prompt (~500 tok)"],
        rows=[
            ["TTFT (ms)", f"{metrics_short.ttft_ms:.1f}", f"{metrics_long.ttft_ms:.1f}"],
            ["Total time (ms)", f"{metrics_short.total_time_ms:.1f}", f"{metrics_long.total_time_ms:.1f}"],
            ["Tokens generated", metrics_short.tokens_generated, metrics_long.tokens_generated],
            ["Tokens/sec", f"{metrics_short.tokens_per_sec:.1f}", f"{metrics_long.tokens_per_sec:.1f}"],
            ["GPU mem (MiB)", f"{gpu_during['used_mem_mb']:.0f}", f"{gpu_after['used_mem_mb']:.0f}"],
        ],
        title="Lab 1 Results: Short vs Long Prompt",
    )

    # ── Key observations ──────────────────────────────────────
    ttft_ratio = metrics_long.ttft_ms / metrics_short.ttft_ms if metrics_short.ttft_ms > 0 else 0
    print("Key Observations:")
    print(f"  - TTFT increased by {ttft_ratio:.1f}x with ~10x more input tokens")
    print(f"    (TTFT grows with prompt length because of prefill computation)")
    print(f"  - GPU memory usage: {gpu_during['used_mem_mb']:.0f} MiB → {gpu_after['used_mem_mb']:.0f} MiB")
    print(f"    (More input tokens → larger KV cache → more memory)")
    print(f"  - Model: {model}")
    print(f"  - GPU: {gpu_before['gpu_name']}")

    # ── Save results ──────────────────────────────────────────
    results = {
        "lab": "lab1_deploy",
        "model": model,
        "gpu": gpu_before,
        "short_prompt": {
            "approx_tokens": 50,
            "ttft_ms": metrics_short.ttft_ms,
            "total_ms": metrics_short.total_time_ms,
            "tokens_generated": metrics_short.tokens_generated,
            "tokens_per_sec": metrics_short.tokens_per_sec,
        },
        "long_prompt": {
            "approx_tokens": 500,
            "ttft_ms": metrics_long.ttft_ms,
            "total_ms": metrics_long.total_time_ms,
            "tokens_generated": metrics_long.tokens_generated,
            "tokens_per_sec": metrics_long.tokens_per_sec,
        },
    }
    save_results(results, "lab1_results.json")
    print("\nLab 1 complete!")


if __name__ == "__main__":
    run_lab1()
