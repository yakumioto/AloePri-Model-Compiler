use crate::{
    error::Result,
    model::{ArchitectureRegistry, ModelArtifact, ModelSpec},
    plan::{PlanDraft, TransformConfig},
};

pub struct Compiler<R> {
    registry: R,
}

impl<R> Compiler<R> {
    pub fn new(registry: R) -> Self {
        Self { registry }
    }
}

impl<R: ArchitectureRegistry> Compiler<R> {
    pub fn inspect<'a>(&self, artifact: &'a dyn ModelArtifact) -> &'a ModelSpec {
        artifact.model_spec()
    }

    pub fn plan_draft(
        &self,
        artifact: &dyn ModelArtifact,
        config: &TransformConfig,
    ) -> Result<PlanDraft> {
        let adapter = self.registry.detect(artifact)?;
        adapter.build_plan(artifact, config)
    }
}
