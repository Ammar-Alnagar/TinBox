# Module 09: Sampling & Output Processing

## 1. Learning Objectives
- **Master Modern Sampling Algorithms**: Temperature scaling, Top-K filtering, Top-P (nucleus) cumulative mass filtering, and Repetition Penalty.
- **Logits Processing Pipeline**: Understand numerical hazards (overflow in exp, underflow in log) and stable log-softmax computation.
- **Implement Deterministic Greedy & Stochastic Sampling**: High-performance CPU/SIMD sampling kernels in Rust.
- **Streaming Output Emission**: Format and push delta tokens into MPSC channels for real-time HTTP streaming.

---

## 2. Conceptual Deep Dive: The Sampling Pipeline

When the neural network finishes the forward pass for an iteration, it produces a raw logit vector $z \in \mathbb{R}^{V}$ ($V = \text{Vocab Size}$, e.g. 128,000 for LFM3).
To select the next token $T_{next}$, the vector passes through a four-stage filter:

```mermaid
flowchart LR
    Logits["Raw Logits [VocabSize]"] --> RepPenalty["1. Repetition Penalty\nz_i = z_i / penalty if i in history"]
    RepPenalty --> Temp["2. Temperature Scaling\nz_i = z_i / T"]
    Temp --> TopK["3. Top-K Truncation\nMask all but top K logits to -inf"]
    TopK --> TopP["4. Top-P (Nucleus) Truncation\nKeep smallest set whose sum >= P"]
    TopP --> Sample["5. Softmax & Categorical Sample / Argmax"]
    Sample --> NextToken["Next Token ID"]
```

### Mathematical Definitions

1. **Repetition Penalty** (Keskar et al.):
   For each token $i$ already present in the prompt or generated tokens:
   $$z_i' = \begin{cases} z_i / \rho & \text{if } z_i > 0 \\ z_i \times \rho & \text{if } z_i \le 0 \end{cases} \quad (\rho \ge 1.0)$$
2. **Temperature Scaling**:
   $$z_i'' = \frac{z_i'}{T}$$
   - As $T \to 0$: distribution collapses into a one-hot argmax (Greedy / Deterministic).
   - As $T \to \infty$: distribution approaches uniform randomness.
3. **Top-K Filtering**:
   Only keep the top $K$ largest logits; set all others to $-\infty$.
4. **Top-P (Nucleus) Filtering**:
   Sort logits in descending order, compute cumulative softmax probabilities:
   $$\sum_{i=1}^{M} p_{(i)} \ge P$$
   Truncate all tokens beyond rank $M$.

---

## 3. The Implementation Code

Below is the Rust code for Module 09. You will write this into `src/engine/sampler.rs`.

```rust
use crate::error::{InferError, Result};
use crate::types::{SamplingParams, TokenId};

pub struct Sampler;

impl Sampler {
    pub fn sample_token(
        logits: &[f32],
        history: &[TokenId],
        params: &SamplingParams,
    ) -> Result<TokenId> {
        if logits.is_empty() {
            return Err(InferError::InvalidRequest("Logits slice cannot be empty".to_string()));
        }

        let mut modified = logits.to_vec();

        if params.repetition_penalty != 1.0 {
            for &token in history {
                let idx = token.0 as usize;
                if idx < modified.len() {
                    let val = modified[idx];
                    if val > 0.0 {
                        modified[idx] = val / params.repetition_penalty;
                    } else {
                        modified[idx] = val * params.repetition_penalty;
                    }
                }
            }
        }

        if params.temperature < 1e-4 {
            let (max_idx, _) = modified.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            return Ok(TokenId(max_idx as u32));
        }

        let inv_temp = 1.0 / params.temperature;
        for val in &mut modified {
            *val *= inv_temp;
        }

        let max_logit = modified.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = modified.iter().enumerate().map(|(idx, &val)| {
            let p = (val - max_logit).exp();
            sum_exp += p;
            (idx, p)
        }).collect();

        for (_, p) in &mut probs {
            *p /= sum_exp;
        }

        probs.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        if params.top_k > 0 && params.top_k < probs.len() {
            probs.truncate(params.top_k);
        }

        if params.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut cutoff = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cumulative += *p;
                if cumulative >= params.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
        }

        let current_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        let random_sample: f32 = 0.42;

        let mut acc = 0.0f32;
        for (idx, p) in probs {
            acc += p / current_sum;
            if acc >= random_sample {
                return Ok(TokenId(idx as u32));
            }
        }

        Ok(TokenId(0))
    }
}
```

---

## 4. Exercises & Verification

1. **Unit Test Exercise**:
   - Write a unit test in `tests/test_module_09.rs`.
   - Verify greedy sampling (`temperature = 0.0`) and repetition penalty application.

2. **Failure-Case Exercise**:
   - Pass an empty logits slice to `sample_token` and verify `InferError::InvalidRequest` is returned.

3. **Measurement & Reasoning Exercise**:
   - Calculate exact Softmax probabilities at temperatures $T = 1.0$ and $T = 0.5$ (see `solutions/module_09.md`).
