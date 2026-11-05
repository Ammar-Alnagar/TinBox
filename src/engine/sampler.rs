use crate::error::{InferError, Result};
use crate::types::{SamplingParams, TokenId};

pub struct Sampler;

impl Sampler {
    pub fn sample_token(
        logits: &[f32],
        history: &[TokenId],
        params: &SamplingParams,
    ) -> Result<TokenId> {
        if logits.is_empty() {
            return Err(InferError::InvalidRequest("Logits slice cannot be empty".to_string()));
        }

        let mut modified = logits.to_vec();

        if params.repetition_penalty != 1.0 {
            for &token in history {
                let idx = token.0 as usize;
                if idx < modified.len() {
                    let val = modified[idx];
                    if val > 0.0 {
                        modified[idx] = val / params.repetition_penalty;
                    } else {
                        modified[idx] = val * params.repetition_penalty;
                    }
                }
            }
        }

        if params.temperature < 1e-4 {
            let (max_idx, _) = modified.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            return Ok(TokenId(max_idx as u32));
        }

        let inv_temp = 1.0 / params.temperature;
        for val in &mut modified {
            *val *= inv_temp;
        }

        let max_logit = modified.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = modified.iter().enumerate().map(|(idx, &val)| {
            let p = (val - max_logit).exp();
            sum_exp += p;
            (idx, p)
        }).collect();

        for (_, p) in &mut probs {
            *p /= sum_exp;
        }

        probs.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        if params.top_k > 0 && params.top_k < probs.len() {
            probs.truncate(params.top_k);
        }

        if params.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut cutoff = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cumulative += *p;
                if cumulative >= params.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
        }

        let current_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        let random_sample: f32 = 0.42;

        let mut acc = 0.0f32;
        for (idx, p) in probs {
            acc += p / current_sum;
            if acc >= random_sample {
                return Ok(TokenId(idx as u32));
            }
        }

        Ok(TokenId(0))
    }
}
