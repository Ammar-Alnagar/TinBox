use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model_id: String,
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub max_batch_size: usize,
    pub max_model_len: usize,
    pub device: DeviceTarget,
    pub enable_prefix_caching: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceTarget {
    Cpu,
    Cuda(usize),
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_id: "synthetic-lfm3-fp8".to_string(),
            block_size: 16,
            num_gpu_blocks: 1024,
            max_batch_size: 32,
            max_model_len: 4096,
            device: DeviceTarget::Cpu,
            enable_prefix_caching: true,
        }
    }
}
