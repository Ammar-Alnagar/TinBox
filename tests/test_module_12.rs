use candle_core::{Device, Result, Tensor};
use std::time::Duration;

// --- Speculative Decoding ---
pub fn verify_draft(
    draft_tokens: &[u32],
    draft_probs: &[Vec<f32>],
    target_probs: &[Vec<f32>],
    random_samples: &[f32],
) -> Vec<u32> {
    assert_eq!(draft_tokens.len(), draft_probs.len());
    assert_eq!(draft_tokens.len(), target_probs.len());
    assert_eq!(draft_tokens.len(), random_samples.len());

    let mut accepted = Vec::new();

    for i in 0..draft_tokens.len() {
        let token = draft_tokens[i];
        let q_val = draft_probs[i][token as usize];
        let p_val = target_probs[i][token as usize];
        let r = random_samples[i];

        let acceptance_ratio = if q_val > 0.0 {
            (p_val / q_val).min(1.0)
        } else {
            1.0
        };

        if r < acceptance_ratio {
            accepted.push(token);
        } else {
            let actual_vocab = target_probs[i].len();
            let mut adjusted = vec![0.0f32; actual_vocab];
            let mut sum_diff = 0.0f32;

            for v in 0..actual_vocab {
                let diff = (target_probs[i][v] - draft_probs[i][v]).max(0.0);
                adjusted[v] = diff;
                sum_diff += diff;
            }

            if sum_diff > 0.0 {
                let mut acc = 0.0f32;
                let mut chosen = 0u32;
                for (v, &prob) in adjusted.iter().enumerate() {
                    acc += prob / sum_diff;
                    if r <= acc {
                        chosen = v as u32;
                        break;
                    }
                }
                accepted.push(chosen);
            } else {
                accepted.push(token);
            }
            break;
        }
    }

    accepted
}

// --- Tensor Parallelism ---
pub fn tensor_parallel_mlp_simulation(
    x: &Tensor,
    w1: &Tensor,
    w2: &Tensor,
) -> Result<Tensor> {
    let (_d_in, d_hidden) = w1.dims2()?;
    let half_h = d_hidden / 2;

    let w1_gpu0 = w1.narrow(1, 0, half_h)?;
    let w2_gpu0 = w2.narrow(0, 0, half_h)?;

    let w1_gpu1 = w1.narrow(1, half_h, half_h)?;
    let w2_gpu1 = w2.narrow(0, half_h, half_h)?;

    let y0 = x.matmul(&w1_gpu0)?;
    let y1 = x.matmul(&w1_gpu1)?;

    let z0 = y0.matmul(&w2_gpu0)?;
    let z1 = y1.matmul(&w2_gpu1)?;

    z0 + z1
}

// --- Kernel Fusion ---
pub fn unfused_rmsnorm_gemm(x: &Tensor, weight_norm: &Tensor, w_linear: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = (x.sqr()?.mean_keepdim(1)? + eps)?;
    let x_normed = x.broadcast_div(&variance.sqrt()?)?;
    let x_scaled = x_normed.broadcast_mul(weight_norm)?;
    x_scaled.matmul(w_linear)
}

pub fn fused_rmsnorm_gemm(x: &Tensor, weight_norm: &Tensor, w_linear: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = (x.sqr()?.mean_keepdim(1)? + eps)?;
    let inv_rms = variance.sqrt()?.recip()?;
    let x_normed = x.broadcast_mul(&inv_rms)?.broadcast_mul(weight_norm)?;
    x_normed.matmul(w_linear)
}

// --- Metrics ---
#[derive(Debug, Clone, Default)]
pub struct ServingMetrics {
    pub total_prefill_duration: Duration,
    pub total_prefill_tokens: usize,
    pub total_decode_duration: Duration,
    pub total_decode_tokens: usize,
    pub allocated_blocks: usize,
    pub total_blocks: usize,
}

impl ServingMetrics {
    pub fn new(total_blocks: usize) -> Self {
        Self {
            total_blocks,
            ..Default::default()
        }
    }

    pub fn record_ttft(&mut self, duration: Duration, num_tokens: usize) {
        self.total_prefill_duration += duration;
        self.total_prefill_tokens += num_tokens;
        tracing::info!(
            target: "tinbox::metrics",
            ttft_ms = duration.as_secs_f64() * 1000.0,
            tokens = num_tokens,
            "Recorded TTFT"
        );
    }

    pub fn record_tpot(&mut self, duration: Duration, num_tokens: usize) {
        self.total_decode_duration += duration;
        self.total_decode_tokens += num_tokens;
        tracing::info!(
            target: "tinbox::metrics",
            tpot_ms = duration.as_secs_f64() * 1000.0,
            tokens = num_tokens,
            "Recorded TPOT"
        );
    }

    pub fn update_memory_pool(&mut self, allocated: usize) {
        self.allocated_blocks = allocated;
        let utilization = self.block_pool_utilization();
        tracing::info!(
            target: "tinbox::metrics",
            allocated = allocated,
            total = self.total_blocks,
            utilization = utilization,
            "Updated Block Pool Utilization"
        );
    }

    #[inline]
    pub fn block_pool_utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.allocated_blocks as f32 / self.total_blocks as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_draft_perfect_match() {
        let draft_tokens = vec![10u32, 20, 30];
        let p_dist = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0; 21],
            vec![0.0; 31],
        ];
        let mut p_probs = p_dist.clone();
        p_probs[1][20] = 1.0;
        p_probs[2][30] = 1.0;

        let q_probs = p_probs.clone();
        let random_samples = vec![0.1f32, 0.5, 0.9];

        let accepted = verify_draft(&draft_tokens, &q_probs, &p_probs, &random_samples);
        assert_eq!(accepted, draft_tokens);
    }

    #[test]
    fn test_tensor_parallel_simulation() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &dev)?;
        let w1 = Tensor::new(&[[1.0f32, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]], &dev)?;
        let w2 = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &dev)?;

        let single_device = x.matmul(&w1)?.matmul(&w2)?;
        let tp_sim = tensor_parallel_mlp_simulation(&x, &w1, &w2)?;

        let diff = (single_device - tp_sim)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_kernel_fusion_equivalence() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &dev)?;
        let weight_norm = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &dev)?;
        let w_linear = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &dev)?;

        let unfused = unfused_rmsnorm_gemm(&x, &weight_norm, &w_linear, 1e-5)?;
        let fused = fused_rmsnorm_gemm(&x, &weight_norm, &w_linear, 1e-5)?;

        let diff = (unfused - fused)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_serving_metrics_with_fake_clock() {
        let mut metrics = ServingMetrics::new(100);
        assert_eq!(metrics.block_pool_utilization(), 0.0);

        metrics.update_memory_pool(40);
        assert!((metrics.block_pool_utilization() - 0.40).abs() < 1e-6);

        metrics.record_ttft(Duration::from_millis(50), 10);
        assert_eq!(metrics.total_prefill_tokens, 10);
        assert_eq!(metrics.total_prefill_duration, Duration::from_millis(50));

        metrics.record_tpot(Duration::from_millis(20), 1);
        assert_eq!(metrics.total_decode_tokens, 1);
        assert_eq!(metrics.total_decode_duration, Duration::from_millis(20));
    }
}
