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
