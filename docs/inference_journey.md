# The Full Journey of an Inference Request

## Key Terms

**P** = total number of parameters in the model. For Llama 70B, P = 70 billion. This is the sum of every number in every weight matrix (W_q, W_k, W_v, W_o, W_ff1, W_ff2) across all layers.
- P stored in FP16 = P × 2 bytes of GPU memory
- Llama 70B: 70B × 2 bytes = 140 GB of weight memory

**d_model** = embedding dimension. For Llama 70B, d_model = 8,192.

**n_layers** = number of transformer layers. For Llama 70B, n_layers = 80.

**N** = number of input tokens (prompt length). For our example, N = 10,000.

---

## Step 0: Load Model Weights into GPU Memory (once at startup)

All weight matrices loaded from disk → GPU HBM (High Bandwidth Memory). P × 2 bytes = 70B × 2 = **140 GB** sitting in GPU HBM. Stays there permanently. Not reloaded per request.

These weights include W_q, W_k, W_v, W_o, W_ff1, W_ff2 for each of 80 layers. They were learned during training. Together they sum to P = 70 billion parameters.

An H100 has 80 GB HBM. 140 GB won't fit → need **tensor parallelism** (split across 2+ GPUs). With 4-way TP, each GPU holds 35 GB of weights.

---

## Step 1: Tokenize + Embed

```
"The quick brown fox jumped over the..." → token IDs → look up each in embedding table
```

```
(wte): Embedding(50257, 8192)
```

Each token becomes a vector of d_model = 8,192 learned numbers. 10,000 tokens → input matrix of **[10,000 × 8,192]**.

---

## Step 2: PREFILL — All Input Tokens Through Every Layer in Parallel

For each of 80 layers:

### 2a: Create Q, K, V (weight multiplications — setup for attention)

This is the setup step for attention. Before tokens can compare against each other in step 2b, each token needs its Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what information do I carry?"). These are created by multiplying against learned weight matrices.

During prefill, all N = 10,000 tokens do this simultaneously in one matrix multiply:

```
All tokens [10,000 × 8,192] × W_q [8,192 × 8,192] = All Queries  [10,000 × 8,192]
All tokens [10,000 × 8,192] × W_k [8,192 × 8,192] = All Keys     [10,000 × 8,192]
All tokens [10,000 × 8,192] × W_v [8,192 × 8,192] = All Values   [10,000 × 8,192]
```

Each costs 2 × N × d_model² = 2 × 10,000 × 8,192 × 8,192 flops.

The **2** is because every parameter contributes one multiply and one add (multiply-accumulate):

```
output value = a×x + b×y + c×z + ...
               ^^^   ^^^   ^^^       ← multiplies
                  ^^^   ^^^          ← adds
= 2 operations per parameter
```

The GPU reads W_q from memory once and uses it for all 10,000 tokens — highly efficient.

**The constraint:** To USE a weight matrix, you must STREAM it from GPU HBM (where it resides) into the compute cores (SMs). The weights are already in GPU memory — they don't get reloaded from disk. But they must flow through the memory bus every time they're used. That streaming is bounded by HBM bandwidth. With tensor parallelism, each GPU only streams its shard:

```
Without TP: P × 2 bytes / mem_bandwidth = 140 GB / 3.35 TB/s ≈ 42 ms on one H100
With 4-way TP: (P/4) × 2 bytes / mem_bandwidth = 35 GB / 3.35 TB/s ≈ 10 ms per GPU
```

This is the flat line on the memory bandwidth vs flops graph — it's the same streaming cost whether you process 1 token or 10,000 tokens. What changes is how much useful math you do while streaming those weights:

- **Prefill (10,000 tokens):** compute time = 2 × P × 10,000 / hardware_flops → HUGE → compute dominates → **compute-bound**
- **Decode (1 token):** compute time = 2 × P × 1 / hardware_flops → tiny → streaming dominates → **memory-bandwidth-bound**

Now the tokens are ready to talk to each other →

### 2b: Attention Scores (token-to-token — NO weights, pure computation)

Every token's Query scores against every token's Key. No weight matrices involved — just tokens comparing to each other:

```
Q [10,000 × 8,192] × K^T [8,192 × 10,000] = Attention scores [10,000 × 10,000]
```

100 million scores. Each cell answers: "how much should token i attend to token j?" This is the **O(N²) quadratic cost**. Double the prompt length → 4x the attention compute.

This cost is NOT part of the 2 × P formula because it doesn't involve model weights (P). It's a separate, sequence-length-dependent cost.

### 2c: Apply Attention to Values (no weights)

The attention scores from 2b determine how much of each token's information to pull:

```
Attention scores [10,000 × 10,000] × V [10,000 × 8,192] = Output [10,000 × 8,192]
```

Each token gets a weighted combination of all other tokens' Values, based on relevance from 2b.

### 2d: Output Projection (weight multiplication — all tokens at once)

Multi-head attention outputs projected back to a single representation:

```
Output [10,000 × 8,192] × W_o [8,192 × 8,192] = Projected [10,000 × 8,192]
```

Cost: 2 × N × d_model² flops. Read W_o once, use for all 10,000 tokens. Same constraint as 2a — weight read bounded by memory bandwidth, but compute dominates at this batch size.

### 2e: Feed-Forward Network (weight multiplications — all tokens at once)

Each token passes through two large weight matrices independently. This is where the model "thinks" — refining each token's representation:

```
Projected [10,000 × 8,192]  × W_ff1 [8,192 × 32,768] = Hidden [10,000 × 32,768]
Hidden    [10,000 × 32,768] × W_ff2 [32,768 × 8,192]  = Output [10,000 × 8,192]
```

Cost: 2 × N × d_model × 4 × d_model flops EACH. These are the biggest weight matrices — the feed-forward is roughly 2/3 of total parameters P. Read each once, use for all 10,000 tokens.

### 2f: Store K and V in KV Cache

The Keys and Values from step 2a get saved in GPU memory for decode to use later. This is the KV cache being built.

**Repeat 2a–2f for all 80 layers.**

### Prefill Summary

- **Weight reads:** W_q, W_k, W_v, W_o, W_ff1, W_ff2 × 80 layers = all P parameters = 140 GB read from memory
- **Weight compute:** 2 × P × N = 2 × 70B × 10,000 flops. Massive. GPU saturated.
- **Attention compute:** N² × d_model × n_layers (separate from weight cost). Quadratic in sequence length.
- **Result: COMPUTE BOUND** — the rising line (2 × P × N / hardware_flops) dominates the flat line (P × 2 bytes / mem_bandwidth)

**End of prefill:** First output token is produced. This moment = **TTFT (Time to First Token).**

---

## Step 3: DECODE — One Token at a Time

Now only ONE new token goes through all 80 layers:

### 3a: Create Q, K, V (weight multiplications — setup, but for just one token)

```
New token [1 × 8,192] × W_q [8,192 × 8,192] = One Query  [1 × 8,192]
New token [1 × 8,192] × W_k [8,192 × 8,192] = One Key    [1 × 8,192]
New token [1 × 8,192] × W_v [8,192 × 8,192] = One Value  [1 × 8,192]
```

Cost: 2 × 1 × d_model² flops each. Same weight matrices streamed from HBM as prefill — all P parameters flow through the memory bus. But used for just ONE token. GPU finishes the math instantly, sits idle waiting for next weight matrix to stream through. Same constraint as prefill, opposite outcome — now memory bandwidth dominates.

### 3b: Attention Against KV Cache (one token queries all stored Keys)

The new token's Query compares against ALL previously cached Keys:

```
Query [1 × 8,192] × All cached Keys^T [8,192 × 10,001] = Scores [1 × 10,001]
```

Not quadratic anymore — just one row of scores. But must READ the entire KV cache from GPU HBM. Longer context = bigger cache to read = slower decode per token. This is an additional memory bandwidth cost on top of the weight streaming. At very long contexts (128K+), KV cache read can dominate even the weight streaming cost — decode becomes KV-cache-bandwidth-bound, not just weight-bandwidth-bound.

### 3c: Apply to Cached Values

```
Scores [1 × 10,001] × All cached Values [10,001 × 8,192] = Output [1 × 8,192]
```

### 3d: Output Projection (weight multiplication — one token)

```
Output [1 × 8,192] × W_o [8,192 × 8,192] = Projected [1 × 8,192]
```

Read W_o (d_model² = 67 million parameters), do tiny math for 1 token.

### 3e: Feed-Forward (weight multiplication — one token)

```
Projected [1 × 8,192] × W_ff1 [8,192 × 32,768] = Hidden [1 × 32,768]
Hidden [1 × 32,768] × W_ff2 [32,768 × 8,192] = Output [1 × 8,192]
```

Read two huge weight matrices (together ~2/3 of P per layer), do tiny math for 1 token.

### 3f: Append New K, V to Cache

Cache grows by one token. Next decode step reads a slightly bigger cache.

**Repeat 3a–3f for all 80 layers. That produces ONE output token. Then repeat for the next token.**

### Decode Summary

- **Weight streaming:** Same P parameters streamed from HBM per step. With 4-way TP, each GPU streams P/4 × 2 bytes.
- **Weight compute:** 2 × P × 1 = 2 × 70B flops. Tiny compared to the streaming cost.
- **KV cache read:** Additional memory bandwidth cost that grows each step. At long contexts, this dominates.
- **Result: MEMORY BANDWIDTH BOUND** — the flat line (P × 2 bytes / mem_bandwidth) dominates the rising line (2 × P × 1 / hardware_flops)

---

## The Key Insight: Arithmetic Intensity

The reason prefill and decode behave differently comes down to one ratio:

```
Arithmetic Intensity = FLOPs / Bytes moved
                     = (2 × P × tokens_processed) / (P × 2 bytes)
                     = tokens_processed
```

- **Prefill (10,000 tokens):** AI = 10,000 → high → compute-bound
- **Decode (1 token):** AI = 1 → low → bandwidth-bound

**The elite-level summary:**

> Prefill has high arithmetic intensity (many FLOPs per byte moved), so it's compute-bound. Decode has low arithmetic intensity (few FLOPs per byte moved), so it's bandwidth-bound.

That's the entire reason prefill vs decode behaves differently — in one sentence.

## The Key Insight

Prefill and decode read the same 140 GB of weights (all P parameters). Prefill does N = 10,000 tokens of useful math while loading them. Decode does 1 token of useful math while loading them. **Same cost, 10,000x less utilization.**

That's why decode is the bottleneck, and why **batching multiple decode requests together helps** — it puts more useful math behind each weight read, moving from memory-bound back toward compute-bound.

---

## Prefill vs Decode Side by Side

```
PREFILL (N = 10,000 tokens):
  Weight streaming: P × 2 bytes / bandwidth           ← flat line (same as decode)
  Weight compute:   2 × P × 10,000 / hardware_flops   ← rising line (DOMINATES)
  Attention:        N² per layer per head               ← additional quadratic cost
  Arithmetic intensity: 10,000 → HIGH
  = COMPUTE BOUND

DECODE (1 token, repeated):
  Weight streaming: P × 2 bytes / bandwidth            ← flat line (DOMINATES)
  Weight compute:   2 × P × 1 / hardware_flops         ← rising line (tiny)
  KV cache read:    grows with context length           ← additional bandwidth cost
  Arithmetic intensity: 1 → LOW
  = MEMORY BANDWIDTH BOUND
```

---

## References

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Inference Explained (Video)](https://www.youtube.com/watch?v=mYRqvB1_gRk&t=1165s)
