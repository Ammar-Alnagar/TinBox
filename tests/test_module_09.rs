use tinbox::engine::sampler::Sampler;
use tinbox::error::InferError;
use tinbox::types::{SamplingParams, TokenId};

#[test]
fn test_module_09_greedy_and_repetition() {
    let logits = vec![1.0f32, 5.0, 2.0];
    let params_greedy = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let tok = Sampler::sample_token(&logits, &[], &params_greedy).unwrap();
    assert_eq!(tok, TokenId(1));

    // Repetition penalty on token 1
    let history = vec![TokenId(1)];
    let params_penalized = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 10.0,
        ..Default::default()
    };
    // Score of 1 becomes 5.0 / 10.0 = 0.5. Argmax is now token 2 (score 2.0).
    let tok_pen = Sampler::sample_token(&logits, &history, &params_penalized).unwrap();
    assert_eq!(tok_pen, TokenId(2));
}

#[test]
fn test_module_09_empty_logits_failure_case() {
    let empty_logits: Vec<f32> = vec![];
    let params = SamplingParams::default();
    let res = Sampler::sample_token(&empty_logits, &[], &params);
    assert!(matches!(res, Err(InferError::InvalidRequest(_))));
}
