use crate::error::Result;
use candle_core::{IndexOp, Tensor};

pub struct PagedAttentionRef;

impl PagedAttentionRef {
    pub fn forward_decode(
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &[usize],
        seq_len: usize,
        block_size: usize,
    ) -> Result<Tensor> {
        let (_num_heads, head_dim) = q.dims2()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut gathered_k = Vec::new();
        let mut gathered_v = Vec::new();

        let mut tokens_remaining = seq_len;
        for &block_id in block_table {
            if tokens_remaining == 0 {
                break;
            }
            let valid_tokens_in_block = tokens_remaining.min(block_size);

            let k_block = k_pool.i((block_id, .., ..valid_tokens_in_block, ..))?;
            let v_block = v_pool.i((block_id, .., ..valid_tokens_in_block, ..))?;

            gathered_k.push(k_block);
            gathered_v.push(v_block);

            tokens_remaining -= valid_tokens_in_block;
        }

        let all_k = Tensor::cat(&gathered_k, 1)?;
        let all_v = Tensor::cat(&gathered_v, 1)?;

        let q_scaled = (q * (scale as f64))?;
        let q_3d = q_scaled.unsqueeze(1)?;
        let all_k_t = all_k.transpose(1, 2)?;
        let attn_scores = q_3d.matmul(&all_k_t)?.squeeze(1)?;

        let attn_weights = candle_nn::ops::softmax(&attn_scores, 1)?;

        let weights_3d = attn_weights.unsqueeze(1)?;
        let output = weights_3d.matmul(&all_v)?.squeeze(1)?;

        Ok(output)
    }
}
