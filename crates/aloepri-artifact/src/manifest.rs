use aloepri_core::{
    error::{CompilerError, Result, io_error, json_error},
    plan::{MethodContract, OutputLayout, TransformPlan},
    types::{DType, TensorName, TensorShape},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorManifest {
    pub name: TensorName,
    pub shape: TensorShape,
    pub dtype: DType,
    pub byte_length: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub artifact_version: u32,
    pub method: MethodContract,
    pub architecture: String,
    pub source_fingerprint: String,
    pub plan_hash: String,
    pub output_layout: OutputLayout,
    pub config_blake3: String,
    pub sidecar_blake3: BTreeMap<String, String>,
    pub tensors: Vec<TensorManifest>,
    pub standard_hf_checkpoint: bool,
    pub secret_key_id: Option<String>,
}

impl Manifest {
    pub fn from_plan(
        plan: &TransformPlan,
        config_bytes: &[u8],
        sidecar_blake3: BTreeMap<String, String>,
        tensors: Vec<TensorManifest>,
    ) -> Self {
        Self {
            artifact_version: MANIFEST_VERSION,
            method: plan.method.clone(),
            architecture: plan.architecture.clone(),
            source_fingerprint: plan.source_fingerprint.to_string(),
            plan_hash: plan.plan_hash.to_string(),
            output_layout: plan.output_layout.clone(),
            config_blake3: blake3::hash(config_bytes).to_hex().to_string(),
            sidecar_blake3,
            tensors,
            standard_hf_checkpoint: true,
            secret_key_id: None,
        }
    }

    pub fn write(&self, root: &Path) -> Result<()> {
        let path = root.join("aloepri.json");
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| CompilerError::Invariant(error.to_string()))?;
        fs::write(&path, bytes).map_err(|source| io_error(&path, source))?;
        Ok(())
    }

    pub fn read(root: &Path) -> Result<Self> {
        let path = root.join("aloepri.json");
        let bytes = fs::read(&path).map_err(|source| io_error(&path, source))?;
        crate::header::reject_duplicate_keys(&bytes).map_err(|reason| {
            CompilerError::InvalidArtifact {
                path: path.clone(),
                reason: format!("invalid or duplicate manifest JSON: {reason}"),
            }
        })?;
        let manifest: Self =
            serde_json::from_slice(&bytes).map_err(|source| json_error(&path, source))?;
        if manifest.artifact_version != MANIFEST_VERSION {
            return Err(CompilerError::UnsupportedVersion {
                version: manifest.artifact_version,
            });
        }
        Ok(manifest)
    }
}
