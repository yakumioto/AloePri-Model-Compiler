pub mod backend;
pub mod compiler;
pub mod error;
pub mod executor;
pub mod io;
pub mod memory;
pub mod model;
pub mod plan;
pub mod types;

pub use backend::{
    ArtifactBackend, InspectionReport, OutputLockGuard, ShardSummary, TransformReport,
    TransformRequest, VerificationOutcome,
};
pub use compiler::Compiler;
pub use error::{CompilerError, Result};
pub use executor::{StreamingExecutor, TransformExecutor};
pub use io::{OutputWriter, TensorReader, TensorWriter};
pub use memory::{MemoryBudget, MemoryReservation};
pub use model::{ArchitectureAdapter, ArchitectureRegistry, ModelArtifact, ModelSpec};
pub use plan::{
    MethodContract, Operation, OutputLayout, OutputShard, OutputTensor, PlanDraft, TransformConfig,
    TransformPlan,
};
pub use types::*;
