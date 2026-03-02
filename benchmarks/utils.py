"""Shared utilities for inference benchmarking: timing, GPU monitoring, plotting."""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# vLLM environment configuration
# ---------------------------------------------------------------------------


def configure_vllm_env():
    """Set environment variables for vLLM compatibility on Colab/T4 GPUs.

    The vLLM V1 engine has a CUTLASS DSL bug where the GPU architecture
    string is not passed to the NVVM compiler on Turing GPUs (T4 / sm_75),
    causing engine core initialization to fail.  Falling back to the V0
    engine avoids this entirely.
    """
    os.environ.setdefault("VLLM_USE_V1", "0")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Optional[dict] = None


def load_config(path: str = "configs/benchmark_config.yaml") -> dict:
    """Load YAML config, caching the result."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(path) as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def results_dir() -> Path:
    cfg = load_config()
    p = Path(cfg["output"]["results_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    """Metrics collected from a single inference request."""
    ttft_ms: float = 0.0          # Time to first token (ms)
    total_time_ms: float = 0.0    # Total request duration (ms)
    tokens_generated: int = 0
    prompt_tokens: int = 0
    tokens_per_sec: float = 0.0   # Decode throughput (output tok/s)
    gpu_mem_mb: float = 0.0       # GPU memory snapshot (MiB)


@dataclass
class BenchmarkResult:
    """Aggregated result for one benchmark configuration."""
    label: str = ""
    metrics: list = field(default_factory=list)  # list[RequestMetrics]

    # Convenience aggregates (filled by compute_aggregates)
    avg_ttft_ms: float = 0.0
    avg_total_ms: float = 0.0
    avg_tps: float = 0.0
    total_throughput_tps: float = 0.0
    avg_gpu_mem_mb: float = 0.0

    def compute_aggregates(self):
        if not self.metrics:
            return
        n = len(self.metrics)
        self.avg_ttft_ms = sum(m.ttft_ms for m in self.metrics) / n
        self.avg_total_ms = sum(m.total_time_ms for m in self.metrics) / n
        tps_vals = [m.tokens_per_sec for m in self.metrics if m.tokens_per_sec > 0]
        self.avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0
        total_tokens = sum(m.tokens_generated for m in self.metrics)
        wall_sec = max(m.total_time_ms for m in self.metrics) / 1000.0 if self.metrics else 1.0
        self.total_throughput_tps = total_tokens / wall_sec if wall_sec > 0 else 0.0
        mem_vals = [m.gpu_mem_mb for m in self.metrics if m.gpu_mem_mb > 0]
        self.avg_gpu_mem_mb = sum(mem_vals) / len(mem_vals) if mem_vals else 0.0

    def to_dict(self) -> dict:
        self.compute_aggregates()
        return {
            "label": self.label,
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "avg_total_ms": round(self.avg_total_ms, 2),
            "avg_tps": round(self.avg_tps, 2),
            "total_throughput_tps": round(self.total_throughput_tps, 2),
            "avg_gpu_mem_mb": round(self.avg_gpu_mem_mb, 1),
            "num_requests": len(self.metrics),
            "raw": [asdict(m) for m in self.metrics],
        }


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------


def get_gpu_memory_mb(device_index: int = 0) -> float:
    """Return current GPU memory usage in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={device_index}"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def get_gpu_utilization(device_index: int = 0) -> float:
    """Return GPU utilization percentage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", f"--id={device_index}"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def gpu_info_snapshot(device_index: int = 0) -> dict:
    """Return a dict with GPU name, total memory, used memory, utilization."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", f"--id={device_index}"],
            text=True,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {
            "gpu_name": parts[0],
            "total_mem_mb": float(parts[1]),
            "used_mem_mb": float(parts[2]),
            "utilization_pct": float(parts[3]),
        }
    except Exception:
        return {"gpu_name": "unknown", "total_mem_mb": 0, "used_mem_mb": 0, "utilization_pct": 0}


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# A block of filler text used to construct prompts of a target token length.
# Roughly 1 token per word for English text.
_FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "Meanwhile, the curious cat watches from the window ledge above. "
    "Birds sing in the tall oak trees as the morning sun rises slowly. "
    "A gentle breeze carries the scent of fresh flowers across the garden. "
)


def make_prompt(target_tokens: int, question: str = "Summarize the above text in one sentence.") -> str:
    """Build a prompt of approximately *target_tokens* tokens.

    Uses repeated filler text to reach the target length, then appends a short
    question so the model has something to answer.  Token counts are approximate
    (we estimate ~1.3 tokens per word for English).
    """
    words_per_token = 0.75  # conservative: ~1.3 tokens per word
    target_words = int(target_tokens * words_per_token)
    filler_words = _FILLER.split()
    repetitions = (target_words // len(filler_words)) + 1
    body = " ".join(filler_words * repetitions)[:target_words]
    body = " ".join(body.split())  # normalize whitespace
    return f"{body}\n\n{question}"


def make_system_prompt(target_tokens: int = 2000) -> str:
    """Generate a long system prompt simulating a coding-assistant context."""
    rules = [
        "You are an expert fashion stylist AI assistant for the Aura platform.",
        "Always consider the user's body type, skin tone, and personal style preferences.",
        "Recommend outfits appropriate for the specified occasion and weather conditions.",
        "Provide confidence scores for each recommendation on a scale of 0.0 to 1.0.",
        "Include color theory reasoning when suggesting color combinations.",
        "Consider seasonal trends but prioritize timeless style principles.",
        "When recommending accessories, ensure they complement the main outfit pieces.",
        "Provide alternative options at different price points when possible.",
        "Consider cultural context and appropriateness for the specified region.",
        "Include care instructions for delicate or special fabric recommendations.",
        "Factor in sustainability and ethical sourcing when available.",
        "Suggest layering options for temperature flexibility.",
        "Consider the user's existing wardrobe items for mix-and-match potential.",
        "Provide styling tips specific to each recommended outfit.",
        "Include fabric and material recommendations suited to the climate.",
    ]
    # Repeat rules to reach target token count
    words_per_token = 0.75
    target_words = int(target_tokens * words_per_token)
    base = " ".join(rules)
    repetitions = (target_words // len(base.split())) + 1
    body = " ".join((base + " ") * repetitions).split()[:target_words]
    return " ".join(body)


# ---------------------------------------------------------------------------
# Streaming client helpers
# ---------------------------------------------------------------------------


def stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> RequestMetrics:
    """Send a streaming chat completion and measure TTFT + throughput.

    Uses the openai Python client in streaming mode.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed")

    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1

    end = time.perf_counter()

    ttft = (first_token_time - start) * 1000 if first_token_time else 0.0
    total_ms = (end - start) * 1000
    decode_time = (end - first_token_time) if first_token_time else 0.0
    tps = (token_count / decode_time) if decode_time > 0 else 0.0

    return RequestMetrics(
        ttft_ms=ttft,
        total_time_ms=total_ms,
        tokens_generated=token_count,
        tokens_per_sec=tps,
        gpu_mem_mb=get_gpu_memory_mb(),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_line(
    x_values: list,
    y_values: list,
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    y2_values: list | None = None,
    y2_label: str = "",
):
    """Create a line plot (optionally dual-axis) and save to results/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_values, y_values, "b-o", linewidth=2, markersize=8, label=y_label)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    if y2_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(x_values, y2_values, "r-s", linewidth=2, markersize=8, label=y2_label)
        ax2.set_ylabel(y2_label, fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out = results_dir() / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out}")


def plot_comparison_bars(
    labels: list[str],
    values_a: list[float],
    values_b: list[float],
    label_a: str,
    label_b: str,
    y_label: str,
    title: str,
    filename: str,
):
    """Side-by-side bar chart comparing two configurations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, values_a, width, label=label_a, color="#4285F4")
    ax.bar(x + width / 2, values_b, width, label=label_b, color="#EA4335")
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    out = results_dir() / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out}")


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def save_results(data: dict | list, filename: str):
    """Save benchmark results as JSON."""
    out = results_dir() / filename
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Results saved: {out}")


def print_table(headers: list[str], rows: list[list], title: str = ""):
    """Print a formatted ASCII table."""
    from tabulate import tabulate
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".2f"))
    print()


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def wait_for_server(
    base_url: str,
    timeout: int = 300,
    interval: int = 5,
    stderr_log: str | None = None,
) -> bool:
    """Poll the vLLM server until it responds or timeout is reached.

    Args:
        base_url: The vLLM API base URL (e.g. ``http://localhost:8000/v1``).
        timeout: Maximum seconds to wait.
        interval: Seconds between polling attempts.
        stderr_log: Optional path to the server's stderr log file.  When the
            server fails to start, the last 40 lines of this file are printed
            to help with debugging.
    """
    import urllib.request
    import urllib.error

    models_url = f"{base_url}/models"
    deadline = time.time() + timeout
    print(f"Waiting for server at {base_url} ...")
    while time.time() < deadline:
        try:
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("  Server is ready!")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval)

    print("  ERROR: Server did not become ready within timeout.")
    # Surface the server's stderr log so the user can diagnose the failure.
    log_path = stderr_log or "/tmp/vllm_stderr.log"
    try:
        with open(log_path) as fh:
            lines = fh.readlines()
        tail = lines[-40:] if len(lines) > 40 else lines
        print(f"\n  ── last {len(tail)} lines of {log_path} ──")
        for line in tail:
            print(f"  {line.rstrip()}")
        print(f"  ── end of log ──\n")
    except FileNotFoundError:
        print(f"  (no log file found at {log_path})")
    return False
