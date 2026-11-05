use tinbox::memory::block_allocator::PhysicalBlockId;
use tinbox::memory::prefix_cache::PrefixCache;
use tinbox::types::TokenId;

#[test]
fn test_module_08_prefix_matching_and_reuse() {
    let mut cache = PrefixCache::new(4);
    let common_prefix = vec![TokenId(10), TokenId(20), TokenId(30), TokenId(40)];
    let block_id = PhysicalBlockId(7);

    cache.insert_block(&common_prefix, block_id);

    // Query sequence sharing the exact 4-token prefix
    let query_tokens = vec![TokenId(10), TokenId(20), TokenId(30), TokenId(40), TokenId(50)];
    let (matched_blocks, matched_tokens) = cache.match_prefix(&query_tokens);

    assert_eq!(matched_tokens, 4);
    assert_eq!(matched_blocks, vec![block_id]);
    assert_eq!(cache.hits, 1);
}

#[test]
fn test_module_08_disjoint_prefix_cache_miss() {
    let mut cache = PrefixCache::new(4);
    let prefix = vec![TokenId(1), TokenId(2), TokenId(3), TokenId(4)];
    cache.insert_block(&prefix, PhysicalBlockId(1));

    // Completely disjoint tokens
    let disjoint = vec![TokenId(99), TokenId(100)];
    let (matched_blocks, matched_tokens) = cache.match_prefix(&disjoint);

    assert_eq!(matched_tokens, 0);
    assert!(matched_blocks.is_empty());
}
