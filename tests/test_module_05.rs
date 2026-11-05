use candle_core::{DType, Device};
use tinbox::engine::backend::{CandleBackend, ForwardBatch, ModelBackend};
use tinbox::memory::kv_cache::KVCachePool;
use tinbox::model::config::ModelConfig;
use tinbox::model::loader::SyntheticModelLoader;

#[test]
fn test_module_05_ragged_forward_execution() {
    let dev = Device::Cpu;
    let config = ModelConfig::test_lfm3_config();
    let weights = SyntheticModelLoader::generate_synthetic_weights(&config, &dev).unwrap();
    let mut kv_pool = KVCachePool::new(16, config.num_hidden_layers, config.num_key_value_heads, 16, config.head_dim, DType::F32, &dev).unwrap();
    let mut backend = CandleBackend::new(config.clone(), weights, dev);

    // 2 sequences: seq 1 has 2 tokens (idx 0, 1; last=1), seq 2 has 3 tokens (idx 2, 3, 4; last=4)
    let batch = ForwardBatch {
        input_tokens: vec![1, 2, 3, 4, 5],
        positions: vec![0, 1, 0, 1, 2],
        seq_lens: vec![2, 3],
        last_token_indices: vec![1, 4],
        block_tables: vec![vec![0], vec![1]],
        slot_mapping: vec![0, 1, 16, 17, 18],
        is_decode_only: false,
    };

    let forward_out = backend.forward(&batch, &mut kv_pool).unwrap();
    assert_eq!(forward_out.logits.dims2().unwrap(), (2, config.vocab_size));
}

#[test]
fn test_module_05_empty_batch_failure_case() {
    let dev = Device::Cpu;
    let config = ModelConfig::test_lfm3_config();
    let weights = SyntheticModelLoader::generate_synthetic_weights(&config, &dev).unwrap();
    let mut kv_pool = KVCachePool::new(4, config.num_hidden_layers, config.num_key_value_heads, 16, config.head_dim, DType::F32, &dev).unwrap();
    let mut backend = CandleBackend::new(config, weights, dev);

    let empty_batch = ForwardBatch {
        input_tokens: vec![],
        positions: vec![],
        seq_lens: vec![],
        last_token_indices: vec![],
        block_tables: vec![],
        slot_mapping: vec![],
        is_decode_only: false,
    };

    let res = backend.forward(&empty_batch, &mut kv_pool);
    assert!(res.is_err());
}
