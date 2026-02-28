#!/usr/bin/env python3
"""Lab 6: Prompt Caching Simulation

Goals:
  - Demonstrate prefix caching with a shared long system prompt
  - Measure TTFT with and without prefix caching enabled
  - Show how repeated contexts (like Claude Code iterations) benefit from caching

Usage:
  # Start vLLM with prefix caching enabled:
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --port 8000 \
      --enable-prefix-caching

  # Then run:
  python -m benchmarks.lab6_prefix_caching

  # Or let the script manage servers (tests both modes):
  python -m benchmarks.lab6_prefix_caching --manage-servers
"""

import argparse
import subprocess
import sys
import time

from benchmarks.utils import (
    get_gpu_memory_mb,
    gpu_info_snapshot,
    load_config,
    make_system_prompt,
    plot_comparison_bars,
    plot_line,
    print_table,
    save_results,
    stream_chat_completion,
    wait_for_server,
)

# Different user questions that all share the same system prompt
USER_QUESTIONS = [
    "What outfit would you recommend for a casual brunch in San Francisco?",
    "Suggest a professional look for a tech interview in Austin.",
    "What should I wear to a summer wedding in Miami?",
    "Recommend a cold-weather layered outfit for Chicago in December.",
    "What's a good date night outfit for a nice restaurant in New York?",
    "Suggest athleisure that transitions from gym to coffee shop in LA.",
    "What should I pack for a 3-day business trip to Seattle?",
    "Recommend a beach vacation wardrobe for Maui.",
    "What's an appropriate outfit for a gallery opening in Portland?",
    "Suggest a festival outfit for Coachella that's stylish but practical.",
]


def start_vllm_server(model: str, port: int, enable_prefix_caching: bool) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    mode = "prefix caching ENABLED" if enable_prefix_caching else "prefix caching DISABLED"
    print(f"  Starting server ({mode}): {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  Server stopped.")


def run_with_shared_prefix(
    base_url: str,
    model: str,
    system_prompt: str,
    output_tokens: int,
) -> list[dict]:
    """Send all questions with the same system prompt and measure each."""
    results = []
    for i, question in enumerate(USER_QUESTIONS):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        m = stream_chat_completion(
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=output_tokens,
        )
        result = {
            "request_num": i + 1,
            "question": question[:60],
            "ttft_ms": round(m.ttft_ms, 2),
            "total_ms": round(m.total_time_ms, 2),
            "tokens_generated": m.tokens_generated,
            "tps": round(m.tokens_per_sec, 2),
            "gpu_mem_mb": get_gpu_memory_mb(),
        }
        results.append(result)
        print(f"    Request {i+1:>2}: TTFT={m.ttft_ms:>7.1f}ms, "
              f"Total={m.total_time_ms:>7.1f}ms, TPS={m.tokens_per_sec:.1f}")
    return results


def run_lab6():
    parser = argparse.ArgumentParser(description="Lab 6: Prompt Caching Simulation")
    parser.add_argument("--manage-servers", action="store_true",
                        help="Start/stop vLLM servers automatically for both modes")
    args = parser.parse_args()

    cfg = load_config()
    base_url = cfg["server"]["base_url"]
    port = cfg["server"]["port"]
    model = cfg["default_model"]
    lab_cfg = cfg["lab6"]
    system_prompt_tokens = lab_cfg["system_prompt_tokens"]
    output_tokens = lab_cfg["output_tokens"]

    print("=" * 60)
    print("  Lab 6: Prompt Caching Simulation")
    print("=" * 60)

    gpu_info = gpu_info_snapshot()
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['total_mem_mb']:.0f} MiB total)")
    print(f"  Model: {model}")
    print(f"  System prompt: ~{system_prompt_tokens} tokens (shared across all requests)")
    print(f"  User questions: {len(USER_QUESTIONS)}")

    system_prompt = make_system_prompt(system_prompt_tokens)

    all_results = {}

    if args.manage_servers:
        # ── Test WITHOUT prefix caching ───────────────────────
        print(f"\n{'─' * 50}")
        print("  Mode: Prefix Caching DISABLED")
        print(f"{'─' * 50}")

        proc = start_vllm_server(model, port, enable_prefix_caching=False)
        if wait_for_server(base_url, timeout=300):
            # Warmup
            stream_chat_completion(
                base_url=base_url, model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            results_no_cache = run_with_shared_prefix(
                base_url, model, system_prompt, output_tokens
            )
            all_results["No Caching"] = results_no_cache
        stop_server(proc)
        time.sleep(5)

        # ── Test WITH prefix caching ──────────────────────────
        print(f"\n{'─' * 50}")
        print("  Mode: Prefix Caching ENABLED")
        print(f"{'─' * 50}")

        proc = start_vllm_server(model, port, enable_prefix_caching=True)
        if wait_for_server(base_url, timeout=300):
            # Warmup
            stream_chat_completion(
                base_url=base_url, model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            results_cached = run_with_shared_prefix(
                base_url, model, system_prompt, output_tokens
            )
            all_results["Prefix Caching"] = results_cached
        stop_server(proc)

    else:
        # Server already running — test whatever mode it's in
        print("\n  (Using pre-started server. Run with --manage-servers to test both modes.)")
        if not wait_for_server(base_url):
            print("Server not reachable. Start it first (see usage above).")
            sys.exit(1)

        # Warmup
        stream_chat_completion(
            base_url=base_url, model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )

        print(f"\n{'─' * 50}")
        print("  Running with current server configuration")
        print(f"{'─' * 50}")
        results = run_with_shared_prefix(
            base_url, model, system_prompt, output_tokens
        )
        all_results["Current Config"] = results

        # Show per-request TTFT trend (caching shows drop after request 1)
        ttfts = [r["ttft_ms"] for r in results]
        first_ttft = ttfts[0]
        subsequent_avg = sum(ttfts[1:]) / len(ttfts[1:]) if len(ttfts) > 1 else 0
        print(f"\n  First request TTFT:        {first_ttft:.1f} ms")
        print(f"  Subsequent avg TTFT:       {subsequent_avg:.1f} ms")
        if first_ttft > 0 and subsequent_avg < first_ttft * 0.7:
            print(f"  → Prefix caching appears ACTIVE (TTFT dropped {(1 - subsequent_avg/first_ttft)*100:.0f}%)")
        else:
            print(f"  → Prefix caching appears INACTIVE (TTFT stable)")

    # ── Comparison (if both modes tested) ─────────────────────
    if "No Caching" in all_results and "Prefix Caching" in all_results:
        nc = all_results["No Caching"]
        pc = all_results["Prefix Caching"]

        table_rows = []
        for i in range(min(len(nc), len(pc))):
            table_rows.append([
                i + 1,
                nc[i]["question"][:40],
                f"{nc[i]['ttft_ms']:.1f}",
                f"{pc[i]['ttft_ms']:.1f}",
                f"{((nc[i]['ttft_ms'] - pc[i]['ttft_ms']) / nc[i]['ttft_ms'] * 100):+.0f}%"
                if nc[i]["ttft_ms"] > 0 else "N/A",
            ])

        print_table(
            headers=["#", "Question", "TTFT No Cache (ms)", "TTFT Cached (ms)", "Improvement"],
            rows=table_rows,
            title="Lab 6: Prefix Caching Impact",
        )

        # Focus on requests 2+ (after prefix is cached)
        nc_subsequent = [r["ttft_ms"] for r in nc[1:]]
        pc_subsequent = [r["ttft_ms"] for r in pc[1:]]
        avg_nc = sum(nc_subsequent) / len(nc_subsequent) if nc_subsequent else 0
        avg_pc = sum(pc_subsequent) / len(pc_subsequent) if pc_subsequent else 0
        improvement = (avg_nc - avg_pc) / avg_nc * 100 if avg_nc > 0 else 0

        print("Key Observations:")
        print(f"  - Avg TTFT without caching (requests 2-{len(nc)}): {avg_nc:.1f} ms")
        print(f"  - Avg TTFT with caching (requests 2-{len(pc)}):    {avg_pc:.1f} ms")
        print(f"  - TTFT improvement with prefix caching:        {improvement:.0f}%")
        print(f"  - Shared prefix: ~{system_prompt_tokens} tokens (style rules + context)")
        print(f"  - This is how Claude Code works efficiently:")
        print(f"    Same codebase context, different questions → skip redundant prefill")

        nc_mem = [r["gpu_mem_mb"] for r in nc]
        pc_mem = [r["gpu_mem_mb"] for r in pc]
        avg_mem_nc = sum(nc_mem) / len(nc_mem) if nc_mem else 0
        avg_mem_pc = sum(pc_mem) / len(pc_mem) if pc_mem else 0
        print(f"  - GPU memory: {avg_mem_nc:.0f} MiB (no cache) vs {avg_mem_pc:.0f} MiB (cached)")

        # ── Plots ─────────────────────────────────────────────
        request_nums = list(range(1, min(len(nc), len(pc)) + 1))
        nc_ttfts = [r["ttft_ms"] for r in nc[:len(request_nums)]]
        pc_ttfts = [r["ttft_ms"] for r in pc[:len(request_nums)]]

        plot_line(
            x_values=request_nums,
            y_values=nc_ttfts,
            x_label="Request Number",
            y_label="TTFT — No Cache (ms)",
            title="Lab 6: TTFT per Request (Prefix Caching Impact)",
            filename="lab6_prefix_caching_ttft.png",
            y2_values=pc_ttfts,
            y2_label="TTFT — Cached (ms)",
        )

        req_labels = [f"Req {i+1}" for i in range(min(len(nc), len(pc)))]
        plot_comparison_bars(
            labels=req_labels[:5],  # first 5 for readability
            values_a=nc_ttfts[:5],
            values_b=pc_ttfts[:5],
            label_a="No Caching",
            label_b="Prefix Caching",
            y_label="TTFT (ms)",
            title="Lab 6: TTFT Comparison (First 5 Requests)",
            filename="lab6_prefix_caching_bars.png",
        )

    # ── Single-mode trend plot ────────────────────────────────
    for label, data in all_results.items():
        if label == "Current Config":
            ttfts = [r["ttft_ms"] for r in data]
            plot_line(
                x_values=list(range(1, len(ttfts) + 1)),
                y_values=ttfts,
                x_label="Request Number",
                y_label="TTFT (ms)",
                title=f"Lab 6: TTFT Trend ({label})",
                filename="lab6_ttft_trend.png",
            )

    # ── Save ──────────────────────────────────────────────────
    save_results(
        {
            "lab": "lab6_prefix_caching",
            "model": model,
            "gpu": gpu_info,
            "system_prompt_tokens": system_prompt_tokens,
            "results": all_results,
        },
        "lab6_results.json",
    )
    print("\nLab 6 complete!")


if __name__ == "__main__":
    run_lab6()
