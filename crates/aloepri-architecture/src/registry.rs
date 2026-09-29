use crate::llama::LlamaDenseAdapter;
use aloepri_core::{
    error::{CompilerError, Result},
    model::{ArchitectureAdapter, ArchitectureRegistry, ModelArtifact},
};

pub struct Registry {
    adapters: Vec<Box<dyn ArchitectureAdapter>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            adapters: vec![Box::new(LlamaDenseAdapter)],
        }
    }
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_adapters(adapters: Vec<Box<dyn ArchitectureAdapter>>) -> Self {
        Self { adapters }
    }
}

impl ArchitectureRegistry for Registry {
    fn detect(&self, artifact: &dyn ModelArtifact) -> Result<Box<dyn ArchitectureAdapter>> {
        let matches: Vec<_> = self
            .adapters
            .iter()
            .filter(|adapter| adapter.id() == "llama" && LlamaDenseAdapter::matches(artifact))
            .collect();
        match matches.len() {
            0 => Err(CompilerError::UnsupportedArchitecture {
                reason: format!(
                    "no registered adapter matches {}",
                    artifact.model_spec().architecture
                ),
            }),
            1 => Ok(Box::new(LlamaDenseAdapter)),
            _ => Err(CompilerError::AmbiguousArchitecture {
                matches: matches.iter().map(|adapter| adapter.id().into()).collect(),
            }),
        }
    }
}
