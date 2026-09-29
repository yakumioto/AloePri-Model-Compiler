use aloepri_core::{
    error::{CompilerError, Result},
    executor::{StreamingExecutor, TransformExecutor},
    io::OutputWriter,
    memory::MemoryBudget,
    model::ModelArtifact,
    plan::{MethodContract, Operation, TransformConfig},
    types::ModelFingerprint,
};

/// Identity transformation: byte-for-byte copy with no dtype conversion.
///
/// This is the only place that knows the method contract; the mechanical copy
/// loop stays in the method-agnostic core executor.
#[derive(Default)]
pub struct IdentityExecutor {
    executor: StreamingExecutor,
}

impl IdentityExecutor {
    pub fn requirements(config: &TransformConfig) -> Result<()> {
        if config.method != MethodContract::identity() {
            return Err(CompilerError::Unsupported(
                "only the identity method is implemented".into(),
            ));
        }
        if config.workers != 1 {
            return Err(CompilerError::Unsupported(
                "identity uses a single worker".into(),
            ));
        }
        if config.output_dtype != aloepri_core::types::OutputDType::Preserve {
            return Err(CompilerError::Unsupported(
                "identity only preserves the source dtype".into(),
            ));
        }
        Ok(())
    }
}

impl TransformExecutor for IdentityExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()> {
        Self::requirements(config)
    }

    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        writer: &mut dyn OutputWriter,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint> {
        self.executor
            .execute_operation(artifact, operation, writer, budget)
    }
}
