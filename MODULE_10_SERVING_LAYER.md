# Module 10: Serving Layer (Axum & Tokio)

## 1. Learning Objectives
- **Build an OpenAI-Compatible Web API**: Expose `/v1/chat/completions` supporting streaming delta updates.
- **Master Server-Sent Events (SSE)**: Implement the chunked `text/event-stream` protocol using Axum and `tokio_stream`.
- **Bridge Network Async I/O to Engine Compute**: Connect Axum HTTP handlers to the continuous batching scheduler via channels.
- **Graceful Shutdown & Backpressure**: Handle client disconnections cleanly without leaking GPU memory.
- **Mermaid End-to-End Request Diagrams**: Map the full HTTP request lifecycle.

---

## 2. Conceptual Deep Dive: Streaming over Server-Sent Events (SSE)

### The OpenAI Streaming Protocol
When a client sends:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```
The server responds with `HTTP/1.1 200 OK` and `Content-Type: text/event-stream; charset=utf-8`.
The server streams chunks line-by-line:
```text
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" world"}}]}

data: [DONE]

```

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Axum as Axum Route Handler
    participant Engine as Engine Runtime Channel

    Client->>Axum: POST /v1/chat/completions {stream: true}
    Axum->>Engine: Push Sequence to Queue with mpsc::channel(tx)
    Axum-->>Client: HTTP 200 OK (text/event-stream)

    loop Token Generation
        Engine->>Axum: tx.send(GenerationOutput { text: "Hi" })
        Axum-->>Client: data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n
    end

    Engine->>Axum: tx.send(GenerationOutput { is_finished: true })
    Axum-->>Client: data: [DONE]\n\n
```

---

## 3. The Implementation Code

Below is the Rust code for Module 10. You will write this into `src/serving/server.rs`.

```rust
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::types::{GenerationOutput, RequestId, SamplingParams, Sequence, TokenId};

/// Inbound request matching the OpenAI Chat Completions API format.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Shared application state passed to Axum route handlers.
pub struct AppState {
    /// Inbound queue sender to deliver new sequences to the engine runtime loop.
    pub queue_tx: mpsc::UnboundedSender<Sequence>,
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "TinBox: healthy\n"
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel::<GenerationOutput>();

    // 1. Convert prompt messages into token IDs (dummy tokenizer for demonstration)
    let prompt_tokens: Vec<TokenId> = payload.messages.iter()
        .flat_map(|m| m.content.bytes().map(|b| TokenId(b as u32)))
        .collect();

    let params = SamplingParams {
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.95),
        max_new_tokens: payload.max_tokens.unwrap_or(128),
        ..Default::default()
    };

    let request_id = RequestId(format!("req-{}", std::time::SystemTime::now().elapsed().unwrap().as_nanos()));
    let sequence = Sequence::new(request_id.clone(), prompt_tokens, params, Some(tx));

    // 2. Submit sequence to the inference engine queue
    let _ = state.queue_tx.send(sequence);

    // 3. Transform receiver into an SSE stream of formatted JSON deltas
    let stream = UnboundedReceiverStream::new(rx).map(move |output| {
        if output.is_finished {
            Ok(Event::default().data("[DONE]"))
        } else {
            let chunk = serde_json::json!({
                "id": output.request_id.0,
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {
                        "content": output.text.unwrap_or_default()
                    },
                    "finish_reason": null
                }]
            });
            Ok(Event::default().data(chunk.to_string()))
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

---

## 4. Exercises & Verification

1. **Test the HTTP Server with curl**:
   - Launch the server on `http://127.0.0.1:8000`.
   - Run `curl -N http://127.0.0.1:8000/health`.
   - Submit a streaming request and verify received chunks end with `data: [DONE]`.
