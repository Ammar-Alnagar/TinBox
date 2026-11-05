use crate::error::Result;
use crate::model::config::ModelConfig;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

#[inline]
pub fn dequantize_e4m3_byte(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    let sign_mul = if sign == 1 { -1.0 } else { 1.0 };

    if exp == 0 {
        sign_mul * (mant as f32 / 8.0) * (2.0f32.powi(-6))
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        let normalized_exp = (exp as i32) - 7;
        sign_mul * (1.0 + mant as f32 / 8.0) * 2.0f32.powi(normalized_exp)
    }
}

pub struct FP8LinearWeight {
    pub raw_fp8_bytes: Vec<u8>,
    pub shape: Vec<usize>,
    pub scale: f32,
}

impl FP8LinearWeight {
    pub fn new(raw_fp8_bytes: Vec<u8>, shape: Vec<usize>, scale: f32) -> Self {
        Self { raw_fp8_bytes, shape, scale }
    }

    pub fn to_candle_tensor(&self, device: &Device) -> Result<Tensor> {
        let mut float_data = Vec::with_capacity(self.raw_fp8_bytes.len());
        for &byte in &self.raw_fp8_bytes {
            let float_val = dequantize_e4m3_byte(byte) * self.scale;
            float_data.push(float_val);
        }

        let tensor = Tensor::from_vec(float_data, self.shape.as_slice(), device)?;
        Ok(tensor)
    }
}

pub struct SyntheticModelLoader;

impl SyntheticModelLoader {
    pub fn generate_synthetic_weights(config: &ModelConfig, device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut weights = HashMap::new();

        let emb = Tensor::randn(0.0f32, 0.02f32, (config.vocab_size, config.hidden_size), device)?;
        weights.insert("model.embed_tokens.weight".to_string(), emb);

        for l in 0..config.num_hidden_layers {
            let q = Tensor::randn(0.0f32, 0.02f32, (config.hidden_size, config.num_attention_heads * config.head_dim), device)?;
            let k = Tensor::randn(0.0f32, 0.02f32, (config.hidden_size, config.num_key_value_heads * config.head_dim), device)?;
            let v = Tensor::randn(0.0f32, 0.02f32, (config.hidden_size, config.num_key_value_heads * config.head_dim), device)?;
            let o = Tensor::randn(0.0f32, 0.02f32, (config.num_attention_heads * config.head_dim, config.hidden_size), device)?;

            weights.insert(format!("model.layers.{}.self_attn.q_proj.weight", l), q);
            weights.insert(format!("model.layers.{}.self_attn.k_proj.weight", l), k);
            weights.insert(format!("model.layers.{}.self_attn.v_proj.weight", l), v);
            weights.insert(format!("model.layers.{}.self_attn.o_proj.weight", l), o);

            let norm = Tensor::ones((config.hidden_size,), DType::F32, device)?;
            weights.insert(format!("model.layers.{}.input_layernorm.weight", l), norm.clone());
            weights.insert(format!("model.layers.{}.post_attention_layernorm.weight", l), norm);

            let gate = Tensor::randn(0.0f32, 0.02f32, (config.hidden_size, config.intermediate_size), device)?;
            let down = Tensor::randn(0.0f32, 0.02f32, (config.intermediate_size, config.hidden_size), device)?;
            weights.insert(format!("model.layers.{}.mlp.gate_proj.weight", l), gate);
            weights.insert(format!("model.layers.{}.mlp.down_proj.weight", l), down);
        }

        let head = Tensor::randn(0.0f32, 0.02f32, (config.vocab_size, config.hidden_size), device)?;
        weights.insert("lm_head.weight".to_string(), head);

        Ok(weights)
    }
}
