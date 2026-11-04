use thiserror::Error;

pub type Result<T> = std::result::Result<T, InferError>;

#[derive(Error, Debug)]
pub enum InferError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Memory error: out of physical blocks (requested: {requested}, available: {available})")]
    OutOfMemory {
        requested: usize,
        available: usize,
    },

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Scheduler error: {0}")]
    Scheduler(String),

    #[error("Tensor/Backend error: {0}")]
    Backend(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}
