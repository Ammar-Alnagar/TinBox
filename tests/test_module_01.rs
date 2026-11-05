use tinbox::types::{
    FinishReason, GenerationOutput, RequestId, SamplingParams, Sequence, SequenceStatus, TokenId,
};
use tokio::sync::mpsc;

#[test]
fn test_module_01_lifecycle_completion_and_stop_token() {
    let prompt = vec![TokenId(101), TokenId(102), TokenId(103)];
    let stop_id = TokenId(999);
    let params = SamplingParams {
        max_new_tokens: 3,
        stop_token_ids: vec![stop_id],
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::unbounded_channel::<GenerationOutput>();
    let mut seq = Sequence::new(RequestId("req-1".to_string()), prompt, params, Some(tx));

    assert_eq!(seq.status, SequenceStatus::Waiting);
    assert_eq!(seq.len(), 3);
    assert!(!seq.is_empty());

    // Append 2 tokens
    seq.append_token(TokenId(201));
    seq.emit_token(TokenId(201), None, None);
    assert_eq!(seq.status, SequenceStatus::Waiting);

    seq.append_token(TokenId(202));
    seq.emit_token(TokenId(202), None, None);
    assert_eq!(seq.status, SequenceStatus::Waiting);

    // Append 3rd token -> reaches max_new_tokens
    seq.append_token(TokenId(203));
    assert_eq!(seq.status, SequenceStatus::Finished(FinishReason::Length));
    seq.emit_token(TokenId(203), None, None);

    let out1 = rx.try_recv().unwrap();
    assert_eq!(out1.token_id, TokenId(201));
    assert!(!out1.is_finished);

    let out2 = rx.try_recv().unwrap();
    assert_eq!(out2.token_id, TokenId(202));
    assert!(!out2.is_finished);

    let out3 = rx.try_recv().unwrap();
    assert_eq!(out3.token_id, TokenId(203));
    assert!(out3.is_finished);
    assert_eq!(out3.finish_reason, Some(FinishReason::Length));
}

#[test]
fn test_module_01_stop_token_trigger() {
    let prompt = vec![TokenId(1)];
    let stop_id = TokenId(50);
    let params = SamplingParams {
        max_new_tokens: 10,
        stop_token_ids: vec![stop_id],
        ..Default::default()
    };

    let mut seq = Sequence::new(RequestId("req-stop".to_string()), prompt, params, None);
    seq.append_token(stop_id);
    assert_eq!(seq.status, SequenceStatus::Finished(FinishReason::Stop));
}

#[test]
fn test_module_01_client_disconnect_failure_case() {
    let (tx, rx) = mpsc::unbounded_channel::<GenerationOutput>();
    drop(rx); // Client disconnects

    let seq = Sequence::new(
        RequestId("req-disc".to_string()),
        vec![TokenId(1)],
        SamplingParams::default(),
        Some(tx),
    );

    // Calling emit_token after client drop must not panic
    seq.emit_token(TokenId(2), None, None);
}
