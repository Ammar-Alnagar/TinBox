use crate::memory::block_allocator::PhysicalBlockId;
use crate::types::TokenId;
use std::collections::HashMap;

pub struct RadixNode {
    pub tokens: Vec<TokenId>,
    pub physical_block: PhysicalBlockId,
    pub children: HashMap<TokenId, RadixNode>,
    pub last_accessed: u64,
}

impl RadixNode {
    pub fn new(tokens: Vec<TokenId>, physical_block: PhysicalBlockId) -> Self {
        Self {
            tokens,
            physical_block,
            children: HashMap::new(),
            last_accessed: 0,
        }
    }
}

pub struct PrefixCache {
    pub roots: HashMap<TokenId, RadixNode>,
    pub block_size: usize,
    pub hits: usize,
    pub misses: usize,
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            roots: HashMap::new(),
            block_size,
            hits: 0,
            misses: 0,
        }
    }

    pub fn match_prefix(&mut self, tokens: &[TokenId]) -> (Vec<PhysicalBlockId>, usize) {
        if tokens.is_empty() {
            return (Vec::new(), 0);
        }

        let mut matched_blocks = Vec::new();
        let mut tokens_matched = 0;

        let mut current_map = &mut self.roots;
        let mut token_offset = 0;

        while token_offset < tokens.len() {
            let first_token = tokens[token_offset];
            if let Some(node) = current_map.get_mut(&first_token) {
                let node_len = node.tokens.len();
                if token_offset + node_len <= tokens.len()
                    && &tokens[token_offset..token_offset + node_len] == node.tokens.as_slice()
                {
                    matched_blocks.push(node.physical_block);
                    tokens_matched += node_len;
                    token_offset += node_len;
                    current_map = &mut node.children;
                    continue;
                }
            }
            break;
        }

        if tokens_matched > 0 {
            self.hits += 1;
        } else {
            self.misses += 1;
        }

        (matched_blocks, tokens_matched)
    }

    pub fn insert_block(&mut self, tokens: &[TokenId], block_id: PhysicalBlockId) {
        if tokens.len() != self.block_size {
            return;
        }

        let first_token = tokens[0];
        self.roots.entry(first_token).or_insert_with(|| {
            RadixNode::new(tokens.to_vec(), block_id)
        });
    }
}
