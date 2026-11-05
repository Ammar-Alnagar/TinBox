use candle_core::{Device, Tensor};
use tinbox::engine::paged_attention::PagedAttentionRef;

#[test]
fn test_module_06_paged_attention_noncontiguous_blocks() {
    let dev = Device::Cpu;
    let num_heads = 2;
    let head_dim = 4;
    let num_blocks = 4;
    let block_size = 4;

    let q = Tensor::ones((num_heads, head_dim), candle_core::DType::F32, &dev).unwrap();
    let k_pool = Tensor::zeros((num_blocks, num_heads, block_size, head_dim), candle_core::DType::F32, &dev).unwrap();
    let v_pool = Tensor::ones((num_blocks, num_heads, block_size, head_dim), candle_core::DType::F32, &dev).unwrap();

    // Use physical blocks 1 and 3 for 6 past tokens
    let block_table = vec![1, 3];
    let seq_len = 6;

    let out = PagedAttentionRef::forward_decode(&q, &k_pool, &v_pool, &block_table, seq_len, block_size).unwrap();
    assert_eq!(out.dims2().unwrap(), (num_heads, head_dim));
}

#[test]
fn test_module_06_edge_case_single_token() {
    let dev = Device::Cpu;
    let num_heads = 1;
    let head_dim = 8;
    let num_blocks = 2;
    let block_size = 16;

    let q = Tensor::ones((num_heads, head_dim), candle_core::DType::F32, &dev).unwrap();
    let k_pool = Tensor::ones((num_blocks, num_heads, block_size, head_dim), candle_core::DType::F32, &dev).unwrap();
    let v_pool = Tensor::ones((num_blocks, num_heads, block_size, head_dim), candle_core::DType::F32, &dev).unwrap();

    let block_table = vec![0];
    let seq_len = 1;

    let out = PagedAttentionRef::forward_decode(&q, &k_pool, &v_pool, &block_table, seq_len, block_size).unwrap();
    assert_eq!(out.dims2().unwrap(), (num_heads, head_dim));
}
