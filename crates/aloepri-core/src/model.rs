use crate::{
    error::Result,
    io::TensorReader,
    plan::{PlanDraft, TransformConfig},
    types::{ModelFingerprint, TensorDescriptor, TensorName},
};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub architecture: String,
    pub config: Value,
    pub tensors: Vec<TensorDescriptor>,
    pub aliases: BTreeMap<String, String>,
}

impl ModelSpec {
    pub fn tensor(&self, name: &TensorName) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|tensor| &tensor.name == name)
    }
}

pub trait ModelArtifact: Send + Sync {
    fn root(&self) -> &Path;
    fn config_bytes(&self) -> &[u8];
    fn config(&self) -> &Value;
    fn model_spec(&self) -> &ModelSpec;
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.model_spec().tensors
    }
    fn tensor_reader(&self, name: &TensorName) -> Result<Box<dyn TensorReader>>;
    fn tensor(&self, name: &TensorName) -> Result<Box<dyn TensorReader>> {
        self.tensor_reader(name)
    }
    fn fingerprint(&self) -> Result<ModelFingerprint>;
}

pub trait ArchitectureAdapter: Send + Sync {
    fn id(&self) -> &'static str;
    fn build_plan(
        &self,
        artifact: &dyn ModelArtifact,
        config: &TransformConfig,
    ) -> Result<PlanDraft>;
}

pub trait ArchitectureRegistry: Send + Sync {
    fn detect(&self, artifact: &dyn ModelArtifact) -> Result<Box<dyn ArchitectureAdapter>>;
}
