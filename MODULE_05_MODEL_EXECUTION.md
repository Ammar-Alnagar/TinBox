# Module 05: Model Execution Backend

## 1. Learning Objectives
- **Design an Abstract Execution Backend Trait**: Decouple model mathematical logic from physical compute hardware (Candle CPU vs. CUDA vs. NVIDIA `cuda-rust`).
- **Implement Forward Pass Semantics**: Understand input tensor preparation for mixed prefill and decode sequences in continuous batching.
- **Implement Tri-Hybrid Execution Flow**: Model the layer-by-layer forward pass combining Gated Short Convolutions, Sliding-Window Attention, and Global Attention.
- **Hardware Agnostic Device Management**: Dynamically target CPU or CUDA GPUs using Rust conditional compilation.

---

## 2. Conceptual Deep Dive: Forward Pass in Continuous Batching

In traditional training or static inference, inputs are rectangular matrices: `[BatchSize, SequenceLength]`.
In continuous batching, a single batch contains:
- Sequence A: Prefill with 120 tokens.
- Sequence B: Prefill with 40 tokens.
- Sequence C: Decode with 1 token.
- Sequence D: Decode with 1 token.

### Flattened 1D Ragged Tokens Representation
Instead of padding with zeros, continuous batch engines flatten all tokens into a **1D Ragged Vector**:
$$\text{Tokens} = [T_A^0, T_A^1, \dots, T_A^{119}, T_B^0, \dots, T_B^{39}, T_C, T_D]$$
$$\text{Total Tokens} = 120 + 40 + 1 + 1 = 162$$

Alongside the flattened tokens, the scheduler passes:
1. `seq_lens`: `[120, 40, 1, 1]`
2. `block_tables`: Per-sequence array of physical block IDs.
3. `slot_mapping`: 1D array mapping each of the 162 tokens to its exact physical slot `(BlockId, Offset)` in the `KVCachePool`.

```mermaid
flowchart TD
    subgraph BatchTokens["Flattened Ragged Batch: 162 Tokens"]
        SeqA["Seq A: Tokens 0..119 (Prefill)"]
        SeqB["Seq B: Tokens 120..159 (Prefill)"]
        SeqC["Seq C: Token 160 (Decode)"]
        SeqD["Seq D: Token 161 (Decode)"]
    end

    subgraph SlotMapping["Slot Mapping Vector [162]"]
        SlotA["Slots for Seq A -> KV Pool"]
        SlotB["Slots for Seq B -> KV Pool"]
        SlotC["Slot for Seq C -> KV Pool"]
        SlotD["Slot for Seq D -> KV Pool"]
    end

    subgraph ForwardPass["Model Execution Forward Pass"]
        Embed["Embedding Lookup -> [162, HiddenDim]"]
        Layers["24 Layers: Convs + Sliding Attn + Global Attn"]
        PagedAttn["Paged Attention Kernels Read & Write via SlotMapping"]
        Norm["RMSNorm"]
        LMHead["LM Head Projection"]
    end

    subgraph Logits["Output Logits"]
        ExtractLogits["Extract Last Token Logits for each Sequence (4 Logits total)"]
    end

    BatchTokens --> Embed
    SlotMapping --> Layers
    Embed --> Layers
    Layers --> PagedAttn
    PagedAttn --> Norm
    Norm --> LMHead
    LMHead --> ExtractLogits
```

---

## 3. The Backend Trait Definition

To support CPU development, CUDA acceleration via Candle, and raw NVIDIA `cuda-rust` / Triton kernels without rewriting the scheduler, we define an abstract trait:

```mermaid
classDiagram
    class ModelBackend {
        <<trait>>
        +device() Device
        +forward(batch) ForwardOutput
        +allocate_kv_pool(num_blocks, block_size) KVCachePool
    }

    class CandleCpuBackend {
        +ModelConfig config
        +HashMap weights
        +Device device
    }

    class CudaRustBackend {
        +ModelConfig config
        +CudaStream stream
        +CutileKernels kernels
    }

    ModelBackend <|.. CandleCpuBackend : implements
    ModelBackend <|.. CudaRustBackend : implements
```

---

## 4. The Implementation Code

Below is the code for Module 05. You will write this into `src/engine/backend.rs`.

```rust
use crate::error::{InferError, Result};
use crate::memory::kv_cache::KVCachePool;
use crate::model::config::{LayerType, ModelConfig};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ForwardBatch {
    pub input_tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub seq_lens: Vec<usize>,
    pub last_token_indices: Vec<usize>,
    pub block_tables: Vec<Vec<usize>>,
    pub slot_mapping: Vec<usize>,
    pub is_decode_only: bool,
}

pub struct ForwardOutput {
    pub logits: Tensor,
}

pub trait ModelBackend: Send + Sync {
    fn device(&self) -> &Device;
    fn forward(&mut self, batch: &ForwardBatch, kv_pool: &mut KVCachePool) -> Result<ForwardOutput>;
}

pub struct CandleBackend {
    pub config: ModelConfig,
    pub weights: HashMap<String, Tensor>,
    pub device: Device,
}

impl CandleBackend {
    pub fn new(config: ModelConfig, weights: HashMap<String, Tensor>, device: Device) -> Self {
        Self { config, weights, device }
    }

    fn execute_layers(&self, mut hidden_states: Tensor, _batch: &ForwardBatch) -> Result<Tensor> {
        let num_layers = self.config.num_hidden_layers;
        let default_layer_types = vec![LayerType::FullAttention; num_layers];
        let layer_types = self.config.layer_types.as_ref().unwrap_or(&default_layer_types);

        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            match layer_type {
                LayerType::Conv => {
                    let norm_w = self.weights.get(&format!("model.layers.{}.input_layernorm.weight", layer_idx))
                        .ok_or_else(|| InferError::Model("Norm weight missing".to_string()))?;
                    let normed = hidden_states.broadcast_mul(norm_w)?;
                    hidden_states = (hidden_states + normed)?;
                }
                LayerType::SlidingAttention | LayerType::FullAttention => {
                    let q_w = self.weights.get(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))
                        .ok_or_else(|| InferError::Model("Q weight missing".to_string()))?;
                    let o_w = self.weights.get(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx))
                        .ok_or_else(|| InferError::Model("O weight missing".to_string()))?;

                    let q = hidden_states.matmul(q_w)?;
                    let attn_out = q.matmul(o_w)?;
                    hidden_states = (hidden_states + attn_out)?;
                }
            }

            let gate_w = self.weights.get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
                .ok_or_else(|| InferError::Model("Gate weight missing".to_string()))?;
            let down_w = self.weights.get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
                .ok_or_else(|| InferError::Model("Down weight missing".to_string()))?;

            let mlp_act = candle_nn::ops::silu(&hidden_states.matmul(gate_w)?)?;
            let mlp_out = mlp_act.matmul(down_w)?;
            hidden_states = (hidden_states + mlp_out)?;
        }

        Ok(hidden_states)
    }
}

impl ModelBackend for CandleBackend {
    fn device(&self) -> &Device {
        &self.device
    }

    fn forward(&mut self, batch: &ForwardBatch, _kv_pool: &mut KVCachePool) -> Result<ForwardOutput> {
        let total_tokens = batch.input_tokens.len();
        if total_tokens == 0 {
            return Err(InferError::Backend("Empty batch passed to forward".to_string()));
        }

        let token_tensor = Tensor::from_vec(batch.input_tokens.clone(), (total_tokens,), &self.device)?;

        let emb_w = self.weights.get("model.embed_tokens.weight")
            .ok_or_else(|| InferError::Model("Missing embedding weight".to_string()))?;
        let hidden_states = emb_w.index_select(&token_tensor, 0)?;

        let out_hidden = self.execute_layers(hidden_states, batch)?;

        let last_indices = Tensor::from_vec(
            batch.last_token_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            (batch.last_token_indices.len(),),
            &self.device,
        )?;
        let last_hidden = out_hidden.index_select(&last_indices, 0)?;

        let lm_head_w = self.weights.get("lm_head.weight")
            .ok_or_else(|| InferError::Model("Missing lm_head weight".to_string()))?;
        let logits = last_hidden.matmul(&lm_head_w.t()?)?;

        Ok(ForwardOutput { logits })
    }
}
```

---

## 5. Exercises & Verification

1. **Unit Test Exercise**:
   - Write a unit test in `tests/test_module_05.rs`.
   - Construct a ragged 1D `ForwardBatch` for 2 sequences and verify forward logits shape `[2, vocab_size]`.

2. **Failure-Case Exercise**:
   - Submit an empty batch to `CandleBackend::forward` and verify that `InferError::Backend` is returned.

3. **Measurement & Reasoning Exercise**:
   - Compute wasted compute percentage under static padding vs ragged batching (see `solutions/module_05.md`).
