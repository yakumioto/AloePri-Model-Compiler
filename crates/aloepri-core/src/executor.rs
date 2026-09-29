use crate::{
    error::Result,
    io::{TensorSink, copy_source_to_sink},
    memory::MemoryBudget,
    model::ModelArtifact,
    plan::{Operation, TransformConfig},
};

/// Mechanical execution of one plan operation.
///
/// The compiler owns ordering, budgeting, sinks and checkpointing; an executor
/// only produces the bytes of a single output into the sink it is handed. It
/// never creates files, plans shards, writes manifests or reports completion —
/// the compiler derives completion from the sink itself.
pub trait TransformExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()>;

    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()>;
}

/// The method-agnostic copy executor: read a bounded chunk and hand it to the sink.
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
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()> {
        let input = &operation.inputs[0].descriptor;
        let mut reader = artifact.tensor_reader(&input.name)?;
        copy_source_to_sink(
            reader.as_mut(),
            sink,
            input,
            &operation.output.descriptor,
            budget,
        )
    }
}
