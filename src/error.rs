use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("RKNN runtime library was not found (tried {tried:?})")]
    RknnLibraryNotFound { tried: Vec<PathBuf> },

    #[error("failed to load RKNN symbol {symbol}: {message}")]
    RknnSymbol { symbol: String, message: String },

    #[error("rknn_{operation} returned {code}")]
    RknnCall { operation: &'static str, code: i32 },

    #[error("tensor {name} has shape {shape:?} with {actual} values, expected {expected}")]
    TensorElementCount {
        name: String,
        shape: Vec<usize>,
        actual: usize,
        expected: usize,
    },

    #[error("invalid tensor: {0}")]
    InvalidTensor(String),

    #[error("model input mismatch: expected {expected} tensors, received {actual}")]
    InputCount { expected: usize, actual: usize },

    #[error("model output {name} is missing")]
    MissingOutput { name: String },

    #[error("unexpected model output layout: {0}")]
    UnexpectedLayout(String),

    #[error("ONNX Runtime backend is disabled; enable the `ort` feature")]
    OrtDisabled,

    #[error("ONNX Runtime error: {0}")]
    Ort(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
