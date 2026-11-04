use tinbox::memory::block_allocator::BlockAllocator;
use tinbox::memory::kv_cache::BlockManager;
use tinbox::memory::prefix_cache::PrefixCache;
use tinbox::model::loader::dequantize_e4m3_byte;
use tinbox::types::{FinishReason, RequestId, SamplingParams, Sequence, SequenceStatus, TokenId};
use tinbox::engine::sampler::Sampler;
use tinbox::engine::paged_attention::PagedAttentionRef;
use candle_core::{Device, Tensor};

#[test]
fn test_sequence_lifecycle() {
    let mut params = SamplingParams::default();
    params.max_new_tokens = 3;
    params.stop_token_ids = vec![TokenId(99)];

    let mut seq = Sequence::new(
        RequestId("test-1".to_string()),
        vec![TokenId(1), TokenId(2)],
        params,
        None,
    );

    assert_eq!(seq.len(), 2);
    assert_eq!(seq.status, SequenceStatus::Waiting);

    seq.append_token(TokenId(10));
    assert_eq!(seq.status, SequenceStatus::Waiting);
    assert_eq!(seq.output_tokens.len(), 1);

    seq.append_token(TokenId(99));
    assert_eq!(seq.status, SequenceStatus::Finished(FinishReason::Stop));
}

#[test]
fn test_block_allocator_and_cow() {
    let mut allocator = BlockAllocator::new(4, 16);
    assert_eq!(allocator.num_free_blocks(), 4);

    let b0 = allocator.allocate().unwrap();
    let b1 = allocator.allocate().unwrap();
    assert_eq!(allocator.num_free_blocks(), 2);

    allocator.retain(b0);
    assert_eq!(allocator.get_ref_count(b0), 2);

    allocator.free(b0);
    assert_eq!(allocator.get_ref_count(b0), 1);
    assert_eq!(allocator.num_free_blocks(), 2);

    allocator.free(b0);
    assert_eq!(allocator.num_free_blocks(), 3);

    allocator.free(b1);
    assert_eq!(allocator.num_free_blocks(), 4);
}

#[test]
fn test_block_manager_fork() {
    let mut bm = BlockManager::new(8, 16);
    let seq = Sequence::new(
        RequestId("parent".to_string()),
        vec![TokenId(1); 20],
        SamplingParams::default(),
        None,
    );

    bm.allocate_for_prompt(&seq).unwrap();
    assert_eq!(bm.allocator.num_free_blocks(), 6);

    let child_id = RequestId("child".to_string());
    bm.fork_sequence(&seq.id, child_id.clone()).unwrap();

    let parent_table = bm.block_tables.get(&seq.id).unwrap();
    for &b in &parent_table.physical_blocks {
        assert_eq!(bm.allocator.get_ref_count(b), 2);
    }

    bm.free_sequence(&seq.id);
    assert_eq!(bm.allocator.num_free_blocks(), 6);

    bm.free_sequence(&child_id);
    assert_eq!(bm.allocator.num_free_blocks(), 8);
}

#[test]
fn test_prefix_cache_matching() {
    let mut pc = PrefixCache::new(4);
    let tokens = vec![TokenId(10), TokenId(20), TokenId(30), TokenId(40)];

    let dummy_block = tinbox::memory::block_allocator::PhysicalBlockId(42);
    pc.insert_block(&tokens, dummy_block);

    let query = vec![TokenId(10), TokenId(20), TokenId(30), TokenId(40), TokenId(50)];
    let (matched, count) = pc.match_prefix(&query);

    assert_eq!(count, 4);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0], dummy_block);
    assert_eq!(pc.hits, 1);
}

#[test]
fn test_fp8_dequantization() {
    let one = dequantize_e4m3_byte(0b00111000);
    assert!((one - 1.0).abs() < 1e-6);

    let neg_one = dequantize_e4m3_byte(0b10111000);
    assert!((neg_one - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_sampler_greedy_and_repetition() {
    let logits = vec![1.0, 5.0, 2.0];
    let history = vec![];
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let sample = Sampler::sample_token(&logits, &history, &params).unwrap();
    assert_eq!(sample, TokenId(1));

    let history_rep = vec![TokenId(1)];
    let params_rep = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 10.0,
        ..Default::default()
    };
    let sample_rep = Sampler::sample_token(&logits, &history_rep, &params_rep).unwrap();
    assert_eq!(sample_rep, TokenId(2));
}

#[test]
fn test_paged_attention_reference() {
    let device = Device::Cpu;
    let q = Tensor::ones((2, 4), candle_core::DType::F32, &device).unwrap();
    let k_pool = Tensor::ones((2, 2, 2, 4), candle_core::DType::F32, &device).unwrap();
    let v_pool = Tensor::ones((2, 2, 2, 4), candle_core::DType::F32, &device).unwrap();

    let block_table = vec![0, 1];
    let seq_len = 3;
    let block_size = 2;

    let out = PagedAttentionRef::forward_decode(&q, &k_pool, &v_pool, &block_table, seq_len, block_size).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}
