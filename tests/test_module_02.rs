use tinbox::error::InferError;
use tinbox::memory::block_allocator::BlockAllocator;

#[test]
fn test_module_02_allocator_basic_and_refcounts() {
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
    assert_eq!(allocator.get_ref_count(b0), 0);
    assert_eq!(allocator.num_free_blocks(), 3);

    allocator.free(b1);
    assert_eq!(allocator.num_free_blocks(), 4);
}

#[test]
fn test_module_02_oom_atomicity_failure_case() {
    let mut allocator = BlockAllocator::new(3, 16);
    let _b0 = allocator.allocate().unwrap();
    assert_eq!(allocator.num_free_blocks(), 2);

    // Requesting 3 blocks when only 2 are free must fail without leaking
    let err = allocator.allocate_exact(3);
    match err {
        Err(InferError::OutOfMemory { requested, available }) => {
            assert_eq!(requested, 3);
            assert_eq!(available, 2);
        }
        _ => panic!("Expected OutOfMemory error"),
    }

    assert_eq!(allocator.num_free_blocks(), 2);
}
