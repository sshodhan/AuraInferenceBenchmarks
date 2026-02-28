#!/usr/bin/env python3
"""Lab 2: Measure KV Cache Impact

Goals:
  - Send prompts of increasing length (100 → 4000 tokens)
  - Measure TTFT, total generation time, and GPU memory at each length
  - Visualize the scaling behavior of prefill cost and KV cache memory
  - Observe the quadratic attention cost becoming visible at longer contexts

Usage:
  # Ensure vLLM server is running (see Lab 1), then:
  python -m benchmarks.lab2_kvcache
"""

import sys
import time

from benchmarks.utils import (
    get_gpu_memory_mb,
    gpu_info_snapshot,
    load_config,
    make_prompt,
    plot_line,
    print_table,
    save_results,
    stream_chat_completion,
    wait_for_server,
)


def run_lab2():
    cfg = load_config()
    base_url = cfg["server"]["base_url"]
    model = cfg["default_model"]
    lab_cfg = cfg["lab2"]
    prompt_lengths = lab_cfg["prompt_lengths"]
    output_tokens = lab_cfg["output_tokens"]
    warmup = lab_cfg["warmup_requests"]

    print("=" * 60)
    print("  Lab 2: Measure KV Cache Impact")
    print("=" * 60)

    # ── Check server ──────────────────────────────────────────
    if not wait_for_server(base_url):
        print("Server not reachable. Start it first (see Lab 1).")
        sys.exit(1)

    gpu_info = gpu_info_snapshot()
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['total_mem_mb']:.0f} MiB total)")

    # ── Warmup ────────────────────────────────────────────────
    print(f"\nWarming up with {warmup} requests...")
    for _ in range(warmup):
        stream_chat_completion(
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )

    # ── Benchmark each prompt length ─────────────────────────
    results_data = []
    table_rows = []

    for length in prompt_lengths:
        print(f"\n[Prompt length: ~{length} tokens]")
        prompt = make_prompt(length)
        messages = [{"role": "user", "content": prompt}]

        # Run 3 iterations and average
        ttfts = []
        totals = []
        tps_list = []
        mem_list = []

        for i in range(3):
            m = stream_chat_completion(
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=output_tokens,
            )
            ttfts.append(m.ttft_ms)
            totals.append(m.total_time_ms)
            tps_list.append(m.tokens_per_sec)
            mem_list.append(get_gpu_memory_mb())
            print(f"  Run {i+1}: TTFT={m.ttft_ms:.1f}ms, Total={m.total_time_ms:.1f}ms, "
                  f"TPS={m.tokens_per_sec:.1f}, Mem={mem_list[-1]:.0f}MiB")

        avg_ttft = sum(ttfts) / len(ttfts)
        avg_total = sum(totals) / len(totals)
        avg_tps = sum(tps_list) / len(tps_list)
        avg_mem = sum(mem_list) / len(mem_list)

        row = {
            "prompt_tokens": length,
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_total_ms": round(avg_total, 2),
            "avg_tps": round(avg_tps, 2),
            "avg_gpu_mem_mb": round(avg_mem, 1),
        }
        results_data.append(row)
        table_rows.append([
            length, f"{avg_ttft:.1f}", f"{avg_total:.1f}",
            f"{avg_tps:.1f}", f"{avg_mem:.0f}",
        ])

    # ── Results table ─────────────────────────────────────────
    print_table(
        headers=["Prompt Tokens", "Avg TTFT (ms)", "Avg Total (ms)",
                 "Avg TPS", "GPU Mem (MiB)"],
        rows=table_rows,
        title="Lab 2 Results: KV Cache Scaling",
    )

    # ── Scaling analysis ──────────────────────────────────────
    if len(results_data) >= 2:
        first = results_data[0]
        last = results_data[-1]
        token_ratio = last["prompt_tokens"] / first["prompt_tokens"]
        ttft_ratio = last["avg_ttft_ms"] / first["avg_ttft_ms"] if first["avg_ttft_ms"] > 0 else 0
        mem_delta = last["avg_gpu_mem_mb"] - first["avg_gpu_mem_mb"]

        print("Scaling Analysis:")
        print(f"  - Prompt length increased {token_ratio:.0f}x "
              f"({first['prompt_tokens']} → {last['prompt_tokens']} tokens)")
        print(f"  - TTFT increased {ttft_ratio:.1f}x "
              f"({first['avg_ttft_ms']:.1f}ms → {last['avg_ttft_ms']:.1f}ms)")
        if ttft_ratio > token_ratio * 0.8:
            print(f"    ↳ TTFT scales super-linearly — quadratic attention cost visible!")
        else:
            print(f"    ↳ TTFT scales roughly linearly at these prompt lengths")
        print(f"  - GPU memory increased by {mem_delta:.0f} MiB "
              f"(KV cache growing with context length)")

    # ── Plots ─────────────────────────────────────────────────
    lengths = [r["prompt_tokens"] for r in results_data]
    ttfts = [r["avg_ttft_ms"] for r in results_data]
    mems = [r["avg_gpu_mem_mb"] for r in results_data]

    plot_line(
        x_values=lengths,
        y_values=ttfts,
        x_label="Prompt Length (tokens)",
        y_label="TTFT (ms)",
        title="Lab 2: TTFT vs Prompt Length",
        filename="lab2_ttft_vs_prompt_length.png",
        y2_values=mems,
        y2_label="GPU Memory (MiB)",
    )

    plot_line(
        x_values=lengths,
        y_values=mems,
        x_label="Prompt Length (tokens)",
        y_label="GPU Memory (MiB)",
        title="Lab 2: GPU Memory vs Prompt Length (KV Cache Growth)",
        filename="lab2_memory_vs_prompt_length.png",
    )

    # ── Save ──────────────────────────────────────────────────
    save_results(
        {"lab": "lab2_kvcache", "model": model, "gpu": gpu_info, "results": results_data},
        "lab2_results.json",
    )
    print("\nLab 2 complete!")


if __name__ == "__main__":
    run_lab2()
