use tinbox::memory::kv_cache::BlockManager;
use tinbox::runtime::scheduler::{Scheduler, SchedulerConfig};
use tinbox::types::{RequestId, SamplingParams, Sequence, SequenceStatus, TokenId};

#[test]
fn test_module_07_scheduler_admission_and_prefill_flag() {
    let bm = BlockManager::new(8, 4);
    let config = SchedulerConfig {
        max_batch_size: 4,
        max_num_batched_tokens: 64,
    };
    let mut scheduler = Scheduler::new(config, bm);

    let seq1 = Sequence::new(
        RequestId("s1".to_string()),
        vec![TokenId(1), TokenId(2)],
        SamplingParams::default(),
        None,
    );
    scheduler.add_sequence(seq1);

    let batch = scheduler.schedule().unwrap();
    assert_eq!(batch.sequences.len(), 1);
    assert!(batch.is_prefill);
    assert_eq!(scheduler.running_map.len(), 1);
}

#[test]
fn test_module_07_preemption_on_memory_exhaustion() {
    let bm = BlockManager::new(2, 2);
    let config = SchedulerConfig {
        max_batch_size: 2,
        max_num_batched_tokens: 16,
    };
    let mut scheduler = Scheduler::new(config, bm);

    let seq1 = Sequence::new(
        RequestId("s1".to_string()),
        vec![TokenId(1), TokenId(2), TokenId(3), TokenId(4)],
        SamplingParams::default(),
        None,
    );
    scheduler.add_sequence(seq1);

    let batch1 = scheduler.schedule().unwrap();
    assert_eq!(batch1.sequences.len(), 1);
    assert_eq!(scheduler.block_manager.allocator.num_free_blocks(), 0);

    // Consume all free blocks so that when s1 is preempted it needs more blocks than free
    // s1 prompt is modified so it requires 3 blocks
    scheduler.running_map.get_mut(&RequestId("s1".to_string())).unwrap().prompt_tokens.push(TokenId(99));
    scheduler.running_map.get_mut(&RequestId("s1".to_string())).unwrap().append_token(TokenId(5));

    // When s1 fails append_token, it frees its 2 blocks. But it now needs 5.div_ceil(2) = 3 blocks!
    // Since only 2 blocks are free in the allocator, s1 cannot be readmitted and remains in waiting_queue as Preempted.
    let batch2 = scheduler.schedule().unwrap();
    assert_eq!(batch2.sequences.len(), 0);
    assert_eq!(scheduler.waiting_queue.len(), 1);
    assert_eq!(scheduler.waiting_queue[0].status, SequenceStatus::Preempted);
    assert_eq!(scheduler.running_map.len(), 0);
}
