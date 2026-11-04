use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    #[serde(rename = "conv")]
    Conv,
    #[serde(rename = "sliding_attention")]
    SlidingAttention,
    #[serde(rename = "full_attention")]
    FullAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub conv_l_cache: Option<usize>,
    pub layer_types: Option<Vec<LayerType>>,
    pub norm_eps: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl ModelConfig {
    pub fn test_lfm3_config() -> Self {
        Self {
            architectures: vec!["Lfm3ForCausalLM".to_string()],
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            sliding_window: Some(32),
            conv_l_cache: Some(7),
            layer_types: Some(vec![
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
                LayerType::SlidingAttention,
            ]),
            norm_eps: 1e-5,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}
