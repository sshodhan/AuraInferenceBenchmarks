#!/usr/bin/env python3
"""Lab 3: Batching & Throughput

Goals:
  - Load-test the vLLM server at increasing concurrency levels (1, 4, 8, 16)
  - Measure total throughput (tokens/sec) and per-request latency
  - Observe continuous batching in action
  - Find the throughput plateau and latency spike point

Usage:
  # Ensure vLLM server is running (see Lab 1), then:
  python -m benchmarks.lab3_batching
"""

import asyncio
import json
import sys
import time

import aiohttp

from benchmarks.utils import (
    get_gpu_memory_mb,
    get_gpu_utilization,
    gpu_info_snapshot,
    load_config,
    make_prompt,
    plot_line,
    print_table,
    save_results,
    wait_for_server,
)


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> dict:
    """Send a single streaming request and measure TTFT + throughput."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    async with session.post(f"{url}/chat/completions", json=payload) as resp:
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000 if first_token_time else 0.0
    total_ms = (end - start) * 1000
    decode_time = (end - first_token_time) if first_token_time else 0.001
    tps = token_count / decode_time if decode_time > 0 else 0.0

    return {
        "ttft_ms": ttft,
        "total_ms": total_ms,
        "tokens": token_count,
        "tps": tps,
    }


async def run_concurrent_requests(
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
) -> dict:
    """Run num_requests concurrently (up to concurrency at a time)."""
    prompt = make_prompt(prompt_tokens)
    messages = [{"role": "user", "content": prompt}]

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(session):
        async with semaphore:
            return await send_streaming_request(
                session, base_url, model, messages, output_tokens
            )

    wall_start = time.perf_counter()
    gpu_util_samples = []

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    wall_end = time.perf_counter()
    wall_time_sec = wall_end - wall_start

    total_tokens = sum(r["tokens"] for r in results)
    throughput = total_tokens / wall_time_sec if wall_time_sec > 0 else 0

    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results) if results else 0
    avg_tps = sum(r["tps"] for r in results) / len(results) if results else 0
    avg_total = sum(r["total_ms"] for r in results) / len(results) if results else 0

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_time_sec": round(wall_time_sec, 2),
        "total_tokens": total_tokens,
        "throughput_tps": round(throughput, 2),
        "avg_ttft_ms": round(avg_ttft, 2),
        "avg_per_request_tps": round(avg_tps, 2),
        "avg_total_ms": round(avg_total, 2),
        "gpu_mem_mb": get_gpu_memory_mb(),
        "gpu_util_pct": get_gpu_utilization(),
    }


def run_lab3():
    cfg = load_config()
    base_url = cfg["server"]["base_url"]
    model = cfg["default_model"]
    lab_cfg = cfg["lab3"]

    print("=" * 60)
    print("  Lab 3: Batching & Throughput")
    print("=" * 60)

    if not wait_for_server(base_url):
        print("Server not reachable. Start it first (see Lab 1).")
        sys.exit(1)

    gpu_info = gpu_info_snapshot()
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['total_mem_mb']:.0f} MiB total)")

    # ── Warmup ────────────────────────────────────────────────
    print("\nWarming up...")
    asyncio.run(run_concurrent_requests(
        base_url, model,
        concurrency=1, num_requests=2,
        prompt_tokens=100, output_tokens=20,
    ))

    # ── Benchmark each concurrency level ─────────────────────
    all_results = []
    table_rows = []

    for conc in lab_cfg["concurrency_levels"]:
        print(f"\n[Concurrency: {conc}]")
        result = asyncio.run(run_concurrent_requests(
            base_url, model,
            concurrency=conc,
            num_requests=lab_cfg["requests_per_level"],
            prompt_tokens=lab_cfg["prompt_tokens"],
            output_tokens=lab_cfg["output_tokens"],
        ))
        all_results.append(result)

        print(f"  Throughput:       {result['throughput_tps']:>8.1f} tok/s (total)")
        print(f"  Per-request TPS:  {result['avg_per_request_tps']:>8.1f} tok/s")
        print(f"  Avg TTFT:         {result['avg_ttft_ms']:>8.1f} ms")
        print(f"  Avg total time:   {result['avg_total_ms']:>8.1f} ms")
        print(f"  GPU utilization:  {result['gpu_util_pct']:>8.0f}%")
        print(f"  GPU memory:       {result['gpu_mem_mb']:>8.0f} MiB")

        table_rows.append([
            conc,
            f"{result['throughput_tps']:.1f}",
            f"{result['avg_per_request_tps']:.1f}",
            f"{result['avg_ttft_ms']:.1f}",
            f"{result['avg_total_ms']:.1f}",
            f"{result['gpu_util_pct']:.0f}",
            f"{result['gpu_mem_mb']:.0f}",
        ])

    # ── Results table ─────────────────────────────────────────
    print_table(
        headers=["Concurrency", "Throughput (tok/s)", "Per-req TPS",
                 "Avg TTFT (ms)", "Avg Total (ms)", "GPU Util %", "GPU Mem (MiB)"],
        rows=table_rows,
        title="Lab 3 Results: Batching & Throughput",
    )

    # ── Analysis ──────────────────────────────────────────────
    if len(all_results) >= 2:
        first = all_results[0]
        best_tp = max(all_results, key=lambda r: r["throughput_tps"])
        last = all_results[-1]

        print("Key Observations:")
        print(f"  - Throughput at concurrency=1: {first['throughput_tps']:.1f} tok/s")
        print(f"  - Peak throughput at concurrency={best_tp['concurrency']}: "
              f"{best_tp['throughput_tps']:.1f} tok/s")
        tp_gain = best_tp["throughput_tps"] / first["throughput_tps"] if first["throughput_tps"] > 0 else 0
        print(f"  - Throughput gain: {tp_gain:.1f}x")

        latency_ratio = last["avg_ttft_ms"] / first["avg_ttft_ms"] if first["avg_ttft_ms"] > 0 else 0
        print(f"  - TTFT degradation: {first['avg_ttft_ms']:.0f}ms → {last['avg_ttft_ms']:.0f}ms "
              f"({latency_ratio:.1f}x)")
        print(f"  - This is the throughput vs latency tradeoff you'd manage as TPM")

    # ── Plots ─────────────────────────────────────────────────
    concs = [r["concurrency"] for r in all_results]
    throughputs = [r["throughput_tps"] for r in all_results]
    ttfts = [r["avg_ttft_ms"] for r in all_results]
    per_req_tps = [r["avg_per_request_tps"] for r in all_results]

    plot_line(
        x_values=concs,
        y_values=throughputs,
        x_label="Concurrency",
        y_label="Total Throughput (tok/s)",
        title="Lab 3: Concurrency vs Throughput",
        filename="lab3_concurrency_vs_throughput.png",
        y2_values=ttfts,
        y2_label="Avg TTFT (ms)",
    )

    plot_line(
        x_values=concs,
        y_values=per_req_tps,
        x_label="Concurrency",
        y_label="Per-Request TPS",
        title="Lab 3: Concurrency vs Per-Request Latency",
        filename="lab3_concurrency_vs_per_request_tps.png",
    )

    # ── Save ──────────────────────────────────────────────────
    save_results(
        {"lab": "lab3_batching", "model": model, "gpu": gpu_info, "results": all_results},
        "lab3_results.json",
    )
    print("\nLab 3 complete!")


if __name__ == "__main__":
    run_lab3()
