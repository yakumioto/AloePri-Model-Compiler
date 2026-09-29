use crate::{
    error::Result,
    io::TensorWriter,
    memory::MemoryBudget,
    model::ModelArtifact,
    plan::{Operation, TransformPlan},
    types::{ModelFingerprint, OperationId},
};
use std::collections::BTreeSet;

#[derive(Default)]
pub struct StreamingExecutor;

impl StreamingExecutor {
    pub fn execute(
        &self,
        artifact: &dyn ModelArtifact,
        plan: &TransformPlan,
        writer: &mut dyn TensorWriter,
        budget: &MemoryBudget,
        completed: &BTreeSet<OperationId>,
    ) -> Result<Vec<(OperationId, ModelFingerprint)>> {
        let mut hashes = Vec::new();
        for operation in &plan.operations {
            if completed.contains(&operation.id) {
                continue;
            }
            let hash = self.execute_operation(artifact, operation, writer, budget)?;
            hashes.push((operation.id, hash));
        }
        writer.sync()?;
        Ok(hashes)
    }

    pub fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        writer: &mut dyn TensorWriter,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint> {
        let mut reader = artifact.tensor_reader(&operation.tensor)?;
        writer.write_tensor(&operation.source, reader.as_mut(), budget)
    }
}
