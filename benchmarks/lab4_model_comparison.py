#!/usr/bin/env python3
"""Lab 4: Compare Model Sizes

Goals:
  - Benchmark two model sizes (0.5B vs 1.5B) on identical workloads
  - Compare TTFT, memory usage, and throughput
  - Understand the model-size tradeoff: speed/capacity vs quality

Usage:
  # This lab starts and stops vLLM servers automatically.
  # Make sure no other vLLM server is running on port 8000.
  python -m benchmarks.lab4_model_comparison

  # Or, run against a pre-started server (one model at a time):
  python -m benchmarks.lab4_model_comparison --server-running
"""

import argparse
import subprocess
import sys
import time

from benchmarks.utils import (
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


def benchmark_model(base_url: str, model: str, prompt_lengths: list[int], output_tokens: int) -> list[dict]:
    """Run the KV-cache-style benchmark for a single model."""
    results = []
    for length in prompt_lengths:
        prompt = make_prompt(length)
        messages = [{"role": "user", "content": prompt}]

        # 3 runs per length, take averages
        ttfts, totals, tps_list = [], [], []
        for _ in range(3):
            m = stream_chat_completion(
                base_url=base_url, model=model,
                messages=messages, max_tokens=output_tokens,
            )
            ttfts.append(m.ttft_ms)
            totals.append(m.total_time_ms)
            tps_list.append(m.tokens_per_sec)

        results.append({
            "prompt_tokens": length,
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2),
            "avg_total_ms": round(sum(totals) / len(totals), 2),
            "avg_tps": round(sum(tps_list) / len(tps_list), 2),
            "gpu_mem_mb": get_gpu_memory_mb(),
        })
        print(f"    {length:>5} tokens → TTFT={results[-1]['avg_ttft_ms']:.1f}ms, "
              f"TPS={results[-1]['avg_tps']:.1f}, Mem={results[-1]['gpu_mem_mb']:.0f}MiB")

    return results


def start_vllm_server(model: str, port: int) -> subprocess.Popen:
    """Start a vLLM server as a background process."""
    print(f"  Starting vLLM server for {model}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    """Stop a vLLM server process."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  Server stopped.")


def run_lab4():
    parser = argparse.ArgumentParser(description="Lab 4: Compare Model Sizes")
    parser.add_argument("--server-running", action="store_true",
                        help="Skip server management; use an already-running server")
    args = parser.parse_args()

    cfg = load_config()
    base_url = cfg["server"]["base_url"]
    port = cfg["server"]["port"]
    models_cfg = cfg["models"]
    prompt_lengths = cfg["lab2"]["prompt_lengths"]
    output_tokens = cfg["lab2"]["output_tokens"]

    model_small = models_cfg["small"]["name"]
    model_medium = models_cfg["medium"]["name"]
    label_small = models_cfg["small"]["label"]
    label_medium = models_cfg["medium"]["label"]

    print("=" * 60)
    print("  Lab 4: Compare Model Sizes")
    print("=" * 60)

    gpu_info = gpu_info_snapshot()
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['total_mem_mb']:.0f} MiB total)")
    print(f"  Comparing: {label_small} vs {label_medium}")
    print(f"  Prompt lengths: {prompt_lengths}")

    all_results = {}

    for model_name, label in [(model_small, label_small), (model_medium, label_medium)]:
        print(f"\n{'─' * 50}")
        print(f"  Benchmarking: {label} ({model_name})")
        print(f"{'─' * 50}")

        proc = None
        if not args.server_running:
            proc = start_vllm_server(model_name, port)

        if not wait_for_server(base_url, timeout=300):
            print(f"  ERROR: Server failed to start for {model_name}")
            if proc:
                stop_server(proc)
            continue

        results = benchmark_model(base_url, model_name, prompt_lengths, output_tokens)
        all_results[label] = results

        if proc:
            stop_server(proc)
            time.sleep(5)  # let GPU memory free up

    # ── Comparison table ──────────────────────────────────────
    if len(all_results) == 2:
        labels_list = list(all_results.keys())
        res_a = all_results[labels_list[0]]
        res_b = all_results[labels_list[1]]

        table_rows = []
        for i, length in enumerate(prompt_lengths):
            if i < len(res_a) and i < len(res_b):
                table_rows.append([
                    length,
                    f"{res_a[i]['avg_ttft_ms']:.1f}", f"{res_b[i]['avg_ttft_ms']:.1f}",
                    f"{res_a[i]['avg_tps']:.1f}", f"{res_b[i]['avg_tps']:.1f}",
                    f"{res_a[i]['gpu_mem_mb']:.0f}", f"{res_b[i]['gpu_mem_mb']:.0f}",
                ])

        print_table(
            headers=[
                "Prompt Tok",
                f"TTFT {labels_list[0]}", f"TTFT {labels_list[1]}",
                f"TPS {labels_list[0]}", f"TPS {labels_list[1]}",
                f"Mem {labels_list[0]}", f"Mem {labels_list[1]}",
            ],
            rows=table_rows,
            title="Lab 4: Model Size Comparison",
        )

        # ── Comparison plots ──────────────────────────────────
        pl = [str(l) for l in prompt_lengths[:min(len(res_a), len(res_b))]]
        ttft_a = [r["avg_ttft_ms"] for r in res_a[:len(pl)]]
        ttft_b = [r["avg_ttft_ms"] for r in res_b[:len(pl)]]

        plot_comparison_bars(
            labels=pl,
            values_a=ttft_a,
            values_b=ttft_b,
            label_a=labels_list[0],
            label_b=labels_list[1],
            y_label="TTFT (ms)",
            title="Lab 4: TTFT Comparison by Model Size",
            filename="lab4_ttft_comparison.png",
        )

        mem_a = [r["gpu_mem_mb"] for r in res_a[:len(pl)]]
        mem_b = [r["gpu_mem_mb"] for r in res_b[:len(pl)]]

        plot_comparison_bars(
            labels=pl,
            values_a=mem_a,
            values_b=mem_b,
            label_a=labels_list[0],
            label_b=labels_list[1],
            y_label="GPU Memory (MiB)",
            title="Lab 4: Memory Comparison by Model Size",
            filename="lab4_memory_comparison.png",
        )

        # ── Analysis ──────────────────────────────────────────
        avg_ttft_a = sum(ttft_a) / len(ttft_a)
        avg_ttft_b = sum(ttft_b) / len(ttft_b)
        avg_mem_a = sum(mem_a) / len(mem_a)
        avg_mem_b = sum(mem_b) / len(mem_b)

        print("Key Observations:")
        print(f"  - {labels_list[1]} TTFT is {avg_ttft_b/avg_ttft_a:.1f}x "
              f"slower than {labels_list[0]}")
        print(f"  - {labels_list[1]} uses {avg_mem_b - avg_mem_a:.0f} MiB "
              f"more GPU memory on average")
        print(f"  - Larger model = more memory for weights = less room for KV cache")
        print(f"  - This maps to Anthropic's Haiku (fast) vs Sonnet vs Opus (capable) tradeoffs")

    # ── Save ──────────────────────────────────────────────────
    save_results(
        {"lab": "lab4_model_comparison", "gpu": gpu_info, "results": all_results},
        "lab4_results.json",
    )
    print("\nLab 4 complete!")


if __name__ == "__main__":
    run_lab4()
