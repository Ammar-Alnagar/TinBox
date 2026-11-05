use tinbox::error::InferError;
use tinbox::memory::kv_cache::{BlockManager, BlockTable};
use tinbox::types::{RequestId, SamplingParams, Sequence, TokenId};

#[test]
fn test_module_03_translate_and_cow() {
    let mut bm = BlockManager::new(8, 4); // block_size = 4
    let req = RequestId("seq-1".to_string());
    let prompt = vec![TokenId(1), TokenId(2), TokenId(3), TokenId(4), TokenId(5)];
    let seq = Sequence::new(req.clone(), prompt, SamplingParams::default(), None);

    bm.allocate_for_prompt(&seq).unwrap();
    let (phys_block, _offset) = {
        let table = bm.block_tables.get(&req).unwrap();
        assert_eq!(table.physical_blocks.len(), 2);
        let (pb, off) = table.translate(5, 4).unwrap();
        assert_eq!(pb, table.physical_blocks[1]);
        assert_eq!(off, 1);
        (pb, off)
    };

    // Fork sequence
    let child_req = RequestId("seq-child".to_string());
    bm.fork_sequence(&req, child_req.clone()).unwrap();
    assert_eq!(bm.allocator.get_ref_count(phys_block), 2);

    // Append token to child -> triggers COW
    let (cow_block, cow_offset) = bm.append_token(&child_req, 5).unwrap();
    assert_ne!(cow_block, phys_block);
    assert_eq!(cow_offset, 1);
    assert_eq!(bm.allocator.get_ref_count(phys_block), 1);
}

#[test]
fn test_module_03_empty_prompt_and_out_of_bounds_failure_cases() {
    let mut bm = BlockManager::new(4, 16);
    let req = RequestId("empty".to_string());
    let seq = Sequence::new(req.clone(), vec![], SamplingParams::default(), None);

    let err = bm.allocate_for_prompt(&seq);
    assert!(matches!(err, Err(InferError::InvalidRequest(_))));

    let table = BlockTable::new(req);
    let trans_err = table.translate(0, 16);
    assert!(matches!(trans_err, Err(InferError::Scheduler(_))));
}
