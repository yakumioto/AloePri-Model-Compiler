use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid artifact at {path}: {reason}")]
    InvalidArtifact { path: PathBuf, reason: String },
    #[error("invalid tensor {name}: {reason}")]
    InvalidTensor { name: String, reason: String },
    #[error("missing tensor {name}")]
    MissingTensor { name: String },
    #[error("unsupported dtype {dtype}")]
    UnsupportedDType { dtype: String },
    #[error("unsupported architecture: {reason}")]
    UnsupportedArchitecture { reason: String },
    #[error("ambiguous architecture: {matches:?}")]
    AmbiguousArchitecture { matches: Vec<String> },
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("arithmetic overflow while calculating {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("memory limit exceeded: requested {requested} bytes with {available} bytes available")]
    MemoryLimitExceeded { requested: u64, available: u64 },
    #[error("invalid plan: {reason}")]
    InvalidPlan { reason: String },
    #[error("resume mismatch: {reason}")]
    ResumeMismatch { reason: String },
    #[error("output corrupted: {reason}")]
    OutputCorrupted { reason: String },
    #[error("unsupported schema version {version}")]
    UnsupportedVersion { version: u32 },
    #[error("path escapes the artifact root: {path}")]
    PathEscape { path: PathBuf },
    #[error("output already exists: {path}")]
    AlreadyExists { path: PathBuf },
    #[error("output lock is unavailable: {path}")]
    LockUnavailable { path: PathBuf },
    #[error("an invariant was violated: {0}")]
    Invariant(String),
}

pub type Result<T> = std::result::Result<T, CompilerError>;

pub fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> CompilerError {
    CompilerError::Io {
        path: path.into(),
        source,
    }
}

pub fn json_error(path: impl Into<PathBuf>, source: serde_json::Error) -> CompilerError {
    CompilerError::Json {
        path: path.into(),
        source,
    }
}
