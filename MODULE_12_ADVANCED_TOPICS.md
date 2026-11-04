# Module 12: Advanced Topics & Production Frontiers

## 1. Learning Objectives
- **Explore Speculative Decoding**: Draft-verify speculative pipelines producing 2–4 tokens per forward step.
- **Understand Tensor Parallelism (TP)**: Megatron-LM style row-parallel and column-parallel tensor distribution using NCCL / CUDA IPC.
- **Investigate Kernel Fusion Opportunities**: Fusing RMSNorm + GEMM, and RoPE + QKV projections.
- **Production Observability & Metrics**: Tracing TTFT (Time-To-First-Token), TPOT (Time-Per-Output-Token), and physical block utilization metrics via Prometheus / OpenTelemetry.

---

## 2. Speculative Decoding Architecture

Because LLM generation is memory-bandwidth bound, generating 1 token takes nearly the same time as generating 5 tokens in parallel if the tokens are known in advance!
In **Speculative Decoding**:
1. A small, fast **Draft Model** (e.g. 100M parameter model) generates $K$ candidate tokens sequentially.
2. The large **Target Model** (e.g. 70B parameter model) validates all $K$ tokens in a **single forward step**.
3. If $M \le K$ tokens pass the verification criterion, all $M$ tokens are accepted simultaneously, yielding up to a **$2\times - 3\times$ speedup** without any loss in generation quality!

```mermaid
sequenceDiagram
    autonumber
    participant Draft as Fast Draft Model (Small)
    participant Target as Large Target Model
    participant KV as Paged Block Table

    loop Every Speculative Iteration
        Draft->>Draft: Generate Token 1, 2, 3, 4 sequentially (Takes 10ms)
        Draft->>Target: Propose [T1, T2, T3, T4]
        Target->>Target: Single GEMM Forward Step on all 4 tokens! (Takes 15ms)
        Target->>Target: Verify logits matching distribution
        alt All 4 Tokens Accepted!
            Target->>KV: Commit blocks for all 4 tokens
            Note over Target,KV: 4 tokens emitted in 25ms (6.2ms / token)!
        else Tokens 1..2 Accepted, Token 3 Rejected
            Target->>KV: Roll back block table to Token 2
            Target->>Target: Resample Token 3 correction from target distribution
        end
    end
```

---

## 3. Multi-GPU Tensor Parallelism (Megatron-LM Style)

When a model is too large to fit in a single GPU's VRAM, or to scale memory bandwidth across GPUs, weights are sharded across $N$ GPUs:

```mermaid
flowchart TD
    subgraph ColumnParallel["1. Column Parallel GEMM (QKV Projections)"]
        W_Col1["GPU 0: W[:, 0..D/2]"]
        W_Col2["GPU 1: W[:, D/2..D]"]
    end

    subgraph RowParallel["2. Row Parallel GEMM (Output Projections)"]
        W_Row1["GPU 0: W[0..D/2, :]"]
        W_Row2["GPU 1: W[D/2..D, :]"]
    end

    subgraph AllReduce["NCCL All-Reduce (Sum)"]
        Sum["Y = Y_0 + Y_1 across NVLink (900 GB/s)"]
    end

    InputToken["Input X"] --> W_Col1
    InputToken --> W_Col2
    W_Col1 --> W_Row1
    W_Col2 --> W_Row2
    W_Row1 --> Sum
    W_Row2 --> Sum
    Sum --> OutputToken["Final Layer Output"]
```

---

## 4. Production Metrics & Observability

In production inference runtimes, three primary metrics define service health:

1. **TTFT (Time-To-First-Token)**:
   - Measures prompt prefill latency.
   - Dominated by GEMM compute and prompt length.
   - Optimized by **Prefix Caching** and **Chunked Prefill**.
2. **TPOT (Time-Per-Output-Token)**:
   - Measures inter-token generation latency during decode.
   - Dominated by memory bandwidth ($GEMV$).
   - Optimized by **FP8 weights**, **PagedAttention**, and **Speculative Decoding**.
3. **KV Cache Memory Utilization**:
   $$\text{Block Pool Utilization} = \frac{\text{Allocated Blocks}}{\text{Total Physical Blocks}}$$
   - If utilization exceeds 95%, the scheduler preempts low-priority requests to avoid out-of-memory crashes.
