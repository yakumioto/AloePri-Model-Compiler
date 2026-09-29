use crate::{
    error::Result,
    io::OutputWriter,
    memory::MemoryBudget,
    model::ModelArtifact,
    plan::{Operation, TransformConfig},
    types::ModelFingerprint,
};

/// Mechanical execution of one plan operation.
///
/// The compiler owns ordering, budgeting and checkpointing; an executor only
/// performs the byte-level work of a single operation and reports its digest.
/// Executors do not know model names, tensor naming rules or architectures.
pub trait TransformExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()>;

    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        writer: &mut dyn OutputWriter,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint>;
}

/// The method-agnostic copy executor: read a bounded chunk, write it, hash it.
#[derive(Default)]
pub struct StreamingExecutor;

impl TransformExecutor for StreamingExecutor {
    fn requirements(&self, _config: &TransformConfig) -> Result<()> {
        Ok(())
    }

    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        writer: &mut dyn OutputWriter,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint> {
        let mut reader = artifact.tensor_reader(&operation.tensor)?;
        writer.write_tensor(&operation.source, reader.as_mut(), budget)
    }
}
