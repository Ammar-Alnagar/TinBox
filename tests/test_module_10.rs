use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tinbox::serving::server::{AppState, create_router};
use tokio::sync::mpsc;
use tower::ServiceExt;

#[tokio::test]
async fn test_module_10_health_endpoint() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let state = Arc::new(AppState {
        queue_tx: tx,
    });
    let app = create_router(state);

    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_module_10_malformed_json_failure_case() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let state = Arc::new(AppState {
        queue_tx: tx,
    });
    let app = create_router(state);

    // Malformed JSON (invalid field types)
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"messages": "not an array"}"#))
        .unwrap();

    let response = app.oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}
