use crate::error::{InferError, Result};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalBlockId(pub usize);

#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub id: PhysicalBlockId,
    pub ref_count: usize,
    pub block_size: usize,
    pub prefix_hash: Option<u64>,
}

impl PhysicalBlock {
    pub fn new(id: PhysicalBlockId, block_size: usize) -> Self {
        Self {
            id,
            ref_count: 0,
            block_size,
            prefix_hash: None,
        }
    }

    #[inline]
    pub fn is_free(&self) -> bool {
        self.ref_count == 0
    }
}

pub struct BlockAllocator {
    num_blocks: usize,
    pub block_size: usize,
    blocks: Vec<PhysicalBlock>,
    free_list: VecDeque<PhysicalBlockId>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = VecDeque::with_capacity(num_blocks);

        for i in 0..num_blocks {
            let id = PhysicalBlockId(i);
            blocks.push(PhysicalBlock::new(id, block_size));
            free_list.push_back(id);
        }

        Self {
            num_blocks,
            block_size,
            blocks,
            free_list,
        }
    }

    #[inline]
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn allocate(&mut self) -> Result<PhysicalBlockId> {
        match self.free_list.pop_front() {
            Some(id) => {
                let block = &mut self.blocks[id.0];
                debug_assert_eq!(block.ref_count, 0, "Allocated block must have ref_count == 0");
                block.ref_count = 1;
                block.prefix_hash = None;
                Ok(id)
            }
            None => Err(InferError::OutOfMemory {
                requested: 1,
                available: 0,
            }),
        }
    }

    pub fn allocate_exact(&mut self, count: usize) -> Result<Vec<PhysicalBlockId>> {
        if self.free_list.len() < count {
            return Err(InferError::OutOfMemory {
                requested: count,
                available: self.free_list.len(),
            });
        }

        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            let id = self.allocate()?;
            allocated.push(id);
        }
        Ok(allocated)
    }

    pub fn retain(&mut self, id: PhysicalBlockId) {
        let block = &mut self.blocks[id.0];
        debug_assert!(block.ref_count > 0, "Cannot retain an unallocated block");
        block.ref_count += 1;
    }

    pub fn free(&mut self, id: PhysicalBlockId) {
        let block = &mut self.blocks[id.0];
        debug_assert!(block.ref_count > 0, "Double-free detected on block {:?}", id);

        block.ref_count -= 1;
        if block.ref_count == 0 {
            block.prefix_hash = None;
            self.free_list.push_back(id);
        }
    }

    pub fn free_batch(&mut self, ids: &[PhysicalBlockId]) {
        for &id in ids {
            self.free(id);
        }
    }

    pub fn get_ref_count(&self, id: PhysicalBlockId) -> usize {
        self.blocks[id.0].ref_count
    }
}
