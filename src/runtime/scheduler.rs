use crate::error::Result;
use crate::memory::kv_cache::BlockManager;
use crate::types::{RequestId, Sequence, SequenceStatus};
use std::collections::{HashMap, VecDeque};

pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_num_batched_tokens: usize,
}

pub struct ScheduledBatch {
    pub sequences: Vec<RequestId>,
    pub is_prefill: bool,
}

pub struct Scheduler {
    pub config: SchedulerConfig,
    pub waiting_queue: VecDeque<Sequence>,
    pub running_map: HashMap<RequestId, Sequence>,
    pub block_manager: BlockManager,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, block_manager: BlockManager) -> Self {
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running_map: HashMap::new(),
            block_manager,
        }
    }

    pub fn add_sequence(&mut self, sequence: Sequence) {
        self.waiting_queue.push_back(sequence);
    }

    pub fn schedule(&mut self) -> Result<ScheduledBatch> {
        let mut scheduled_ids = Vec::new();

        let mut preempt_candidate_ids: Vec<RequestId> = self.running_map.keys().cloned().collect();
        let running_ids_to_process: Vec<RequestId> = self.running_map.keys().cloned().collect();

        for req_id in running_ids_to_process {
            let seq = self.running_map.get(&req_id).unwrap();
            let current_len = seq.len();

            match self.block_manager.append_token(&req_id, current_len) {
                Ok(_) => {
                    scheduled_ids.push(req_id);
                }
                Err(_) => {
                    if let Some(victim_id) = preempt_candidate_ids.pop() {
                        if let Some(mut victim_seq) = self.running_map.remove(&victim_id) {
                            victim_seq.status = SequenceStatus::Preempted;
                            self.block_manager.free_sequence(&victim_id);
                            self.waiting_queue.push_front(victim_seq);
                        }
                    }
                }
            }
        }

        while scheduled_ids.len() < self.config.max_batch_size {
            if let Some(waiting_seq) = self.waiting_queue.front() {
                let prompt_blocks_needed = waiting_seq.prompt_tokens.len().div_ceil(self.block_manager.block_size);

                if self.block_manager.allocator.num_free_blocks() >= prompt_blocks_needed {
                    let mut seq = self.waiting_queue.pop_front().unwrap();
                    self.block_manager.allocate_for_prompt(&seq)?;
                    seq.status = SequenceStatus::Running;
                    let id = seq.id.clone();
                    self.running_map.insert(id.clone(), seq);
                    scheduled_ids.push(id);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let is_prefill = scheduled_ids.iter().any(|id| {
            self.running_map.get(id).is_some_and(|s| s.output_tokens.is_empty())
        });

        Ok(ScheduledBatch {
            sequences: scheduled_ids,
            is_prefill,
        })
    }

    pub fn cleanup_finished(&mut self, finished_ids: &[RequestId]) {
        for id in finished_ids {
            self.running_map.remove(id);
            self.block_manager.free_sequence(id);
        }
    }
}
