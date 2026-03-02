# AuraInferenceBenchmarks

Hands-on inference performance labs — deploy models with vLLM, measure TTFT, KV cache scaling, batching throughput, quantization impact, and prefix caching.

By the end you can say: **"I deployed and benchmarked inference myself."**

## Quick Start

### Option A: Google Colab (Easiest)

1. Open [`notebooks/inference_labs.ipynb`](notebooks/inference_labs.ipynb) in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells

### Option B: Local / Cloud GPU

```bash
# Install dependencies
pip install -r requirements.txt

# Start the vLLM server (terminal 1)
./scripts/start_server.sh

# Run labs (terminal 2)
python -m benchmarks.lab1_deploy
python -m benchmarks.lab2_kvcache
python -m benchmarks.lab3_batching
python -m benchmarks.lab6_prefix_caching

# Labs 4 & 5 manage their own servers:
python -m benchmarks.lab4_model_comparison
python -m benchmarks.lab5_quantization
```

## Labs

| Lab | What You Learn | Time |
|-----|---------------|------|
| **Lab 1**: Deploy Model | Model loading, TTFT measurement, GPU memory usage | 1 hr |
| **Lab 2**: KV Cache | TTFT scaling with prompt length, memory growth, quadratic attention cost | 45 min |
| **Lab 3**: Batching | Continuous batching, throughput vs latency tradeoff, GPU saturation | 45 min |
| **Lab 4**: Model Sizes | 0.5B vs 1.5B comparison — speed/capacity vs quality tradeoff | 30 min |
| **Lab 5**: Quantization | FP16 vs INT8 — memory savings, speed changes, quality comparison | 30 min |
| **Lab 6**: Prefix Caching | Shared system prompts, TTFT reduction for repeated contexts | 30 min |

## Project Structure

```
AuraInferenceBenchmarks/
├── benchmarks/
│   ├── utils.py              # Shared: timing, GPU monitoring, plotting
│   ├── lab1_deploy.py        # Lab 1: Deploy & first measurements
│   ├── lab2_kvcache.py       # Lab 2: KV cache scaling
│   ├── lab3_batching.py      # Lab 3: Concurrency & throughput
│   ├── lab4_model_comparison.py  # Lab 4: Model size comparison
│   ├── lab5_quantization.py  # Lab 5: FP16 vs INT8
│   └── lab6_prefix_caching.py    # Lab 6: Prefix caching
├── configs/
│   └── benchmark_config.yaml # Tunable parameters for all labs
├── notebooks/
│   └── inference_labs.ipynb  # Colab-ready notebook (all 6 labs)
├── scripts/
│   ├── start_server.sh       # Start vLLM server
│   └── run_all_labs.sh       # Run all labs sequentially
├── results/                  # Output: JSON data + PNG plots
└── requirements.txt
```

## Configuration

Edit `configs/benchmark_config.yaml` to adjust:
- Model names (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- Prompt lengths for KV cache tests
- Concurrency levels for batching tests
- Output token counts
- Server host/port

## Hardware Requirements

| Platform | GPU | VRAM | Cost |
|----------|-----|------|------|
| Google Colab Free | T4 | 16 GB | $0 |
| Google Colab Pro | T4/A100 | 16-40 GB | $10/mo |
| RunPod | A10G | 24 GB | ~$0.75/hr |
| Lambda Labs | A10G | 24 GB | ~$0.75/hr |
| Local | Any NVIDIA | 16 GB+ | — |

## Theory

| Document | What It Covers |
|----------|---------------|
| [**The Full Journey of an Inference Request**](docs/inference_journey.md) | Step-by-step walkthrough of how a prompt becomes output tokens — tokenization, prefill, decode, KV cache, arithmetic intensity, and why decode is memory-bandwidth-bound |

## Connection to Aura Ecosystem

These labs teach the exact concepts needed to eventually self-host a model for AuraApp:

| Lab Concept | AuraApp Application |
|-------------|-------------------|
| vLLM deployment | Self-host a fine-tuned fashion model instead of Gemini API |
| KV cache & TTFT | Prefill cost for outfit generation with long style-rule prompts |
| Batching throughput | Nightly pipeline generates 40-60K outfits — batching is critical |
| Model size tradeoff | Small/fast for real-time vs larger for nightly batch |
| Quantization | INT8 to cut GPU costs for the nightly pipeline by ~50% |
| Prefix caching | Style rules shared across all requests — slashes nightly compute |
