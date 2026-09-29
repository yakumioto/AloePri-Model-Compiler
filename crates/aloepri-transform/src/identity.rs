use aloepri_core::{
    error::{CompilerError, Result},
    executor::StreamingExecutor,
    io::TensorWriter,
    memory::MemoryBudget,
    model::ModelArtifact,
    plan::{MethodContract, Operation, TransformConfig, TransformPlan},
    types::{ModelFingerprint, OperationId},
};
use std::collections::BTreeSet;

pub struct IdentityExecutor {
    executor: StreamingExecutor,
}

impl Default for IdentityExecutor {
    fn default() -> Self {
        Self {
            executor: StreamingExecutor,
        }
    }
}

impl IdentityExecutor {
    pub fn requirements(config: &TransformConfig) -> Result<()> {
        if config.method != MethodContract::identity() {
            return Err(CompilerError::Unsupported(
                "only identity is implemented".into(),
            ));
        }
        if config.workers != 1 {
            return Err(CompilerError::Unsupported(
                "identity uses one worker".into(),
            ));
        }
        Ok(())
    }

    pub fn execute(
        &self,
        artifact: &dyn ModelArtifact,
        plan: &TransformPlan,
        writer: &mut dyn TensorWriter,
        budget: &MemoryBudget,
        completed: &BTreeSet<OperationId>,
    ) -> Result<Vec<(OperationId, ModelFingerprint)>> {
        Self::requirements(&TransformConfig {
            method: plan.method.clone(),
            output_dtype: aloepri_core::types::OutputDType::Preserve,
            memory_limit: aloepri_core::types::ByteLength(budget.limit()),
            max_shard_size: aloepri_core::types::ByteLength(1),
            workers: 1,
        })?;
        self.executor
            .execute(artifact, plan, writer, budget, completed)
    }

    pub fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        plan: &TransformPlan,
        operation: &Operation,
        writer: &mut dyn TensorWriter,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint> {
        Self::requirements(&TransformConfig {
            method: plan.method.clone(),
            output_dtype: aloepri_core::types::OutputDType::Preserve,
            memory_limit: aloepri_core::types::ByteLength(budget.limit()),
            max_shard_size: aloepri_core::types::ByteLength(1),
            workers: 1,
        })?;
        self.executor
            .execute_operation(artifact, operation, writer, budget)
    }
}
