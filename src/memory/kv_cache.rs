use crate::error::{InferError, Result};
use crate::memory::block_allocator::{BlockAllocator, PhysicalBlockId};
use crate::types::{RequestId, Sequence};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BlockTable {
    pub request_id: RequestId,
    pub physical_blocks: Vec<PhysicalBlockId>,
}

impl BlockTable {
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            physical_blocks: Vec::new(),
        }
    }

    #[inline]
    pub fn append(&mut self, block: PhysicalBlockId) {
        self.physical_blocks.push(block);
    }

    pub fn translate(&self, token_idx: usize, block_size: usize) -> Result<(PhysicalBlockId, usize)> {
        let block_idx = token_idx / block_size;
        let offset = token_idx % block_size;

        self.physical_blocks.get(block_idx)
            .copied()
            .map(|b| (b, offset))
            .ok_or_else(|| InferError::Scheduler(format!(
                "Token index {} (logical block {}) out of bounds in block table of len {}",
                token_idx, block_idx, self.physical_blocks.len()
            )))
    }
}

pub struct KVCachePool {
    pub k_pool: Tensor,
    pub v_pool: Tensor,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub block_size: usize,
    pub head_dim: usize,
    pub device: Device,
}

impl KVCachePool {
    pub fn new(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        block_size: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (num_blocks, num_layers, num_kv_heads, block_size, head_dim);
        let k_pool = Tensor::zeros(shape, dtype, device)?;
        let v_pool = Tensor::zeros(shape, dtype, device)?;

        Ok(Self {
            k_pool,
            v_pool,
            num_blocks,
            num_layers,
            num_kv_heads,
            block_size,
            head_dim,
            device: device.clone(),
        })
    }
}

pub struct BlockManager {
    pub allocator: BlockAllocator,
    pub block_tables: HashMap<RequestId, BlockTable>,
    pub block_size: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks, block_size),
            block_tables: HashMap::new(),
            block_size,
        }
    }

    pub fn allocate_for_prompt(&mut self, sequence: &Sequence) -> Result<()> {
        let num_tokens = sequence.prompt_tokens.len();
        if num_tokens == 0 {
            return Err(InferError::InvalidRequest("Prompt cannot be empty".to_string()));
        }

        let num_blocks_needed = num_tokens.div_ceil(self.block_size);
        let physical_blocks = self.allocator.allocate_exact(num_blocks_needed)?;

        let mut table = BlockTable::new(sequence.id.clone());
        table.physical_blocks = physical_blocks;
        self.block_tables.insert(sequence.id.clone(), table);

        Ok(())
    }

    pub fn append_token(&mut self, request_id: &RequestId, current_seq_len: usize) -> Result<(PhysicalBlockId, usize)> {
        let table = self.block_tables.get_mut(request_id)
            .ok_or_else(|| InferError::Scheduler(format!("No block table found for request {}", request_id)))?;

        let block_idx = current_seq_len / self.block_size;
        let offset = current_seq_len % self.block_size;

        if block_idx >= table.physical_blocks.len() {
            let new_block = self.allocator.allocate()?;
            table.append(new_block);
            return Ok((new_block, offset));
        }

        let physical_block = table.physical_blocks[block_idx];
        if self.allocator.get_ref_count(physical_block) > 1 {
            let new_block = self.allocator.allocate()?;
            self.allocator.free(physical_block);
            table.physical_blocks[block_idx] = new_block;
            return Ok((new_block, offset));
        }

        Ok((physical_block, offset))
    }

    pub fn fork_sequence(&mut self, parent_id: &RequestId, child_id: RequestId) -> Result<()> {
        let parent_table = self.block_tables.get(parent_id)
            .ok_or_else(|| InferError::Scheduler(format!("Parent request {} not found", parent_id)))?;

        let mut child_table = BlockTable::new(child_id.clone());
        for &block in &parent_table.physical_blocks {
            self.allocator.retain(block);
            child_table.append(block);
        }

        self.block_tables.insert(child_id, child_table);
        Ok(())
    }

    pub fn free_sequence(&mut self, request_id: &RequestId) {
        if let Some(table) = self.block_tables.remove(request_id) {
            self.allocator.free_batch(&table.physical_blocks);
        }
    }
}
