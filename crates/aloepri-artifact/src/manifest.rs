use aloepri_core::{
    error::{CompilerError, Result, io_error, json_error},
    plan::{MethodContract, OutputLayout, RuntimeContract, SCHEMA_VERSION, TransformPlan},
    types::{DType, TensorName, TensorShape},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

/// Legacy identity manifest written by v1.
pub const MANIFEST_VERSION: u32 = 1;
/// Legacy token manifest written by v2.
pub const TOKEN_MANIFEST_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorManifest {
    pub name: TensorName,
    pub shape: TensorShape,
    pub dtype: DType,
    pub byte_length: u64,
    pub blake3: String,
}

/// The published artifact descriptor.
///
/// A v3 manifest embeds the full non-secret [`TransformPlan`] so a verifier can
/// recompute the plan/layout relationship instead of trusting a bare hash. The
/// duplicated outer fields are a compatibility projection and must agree with
/// the embedded plan; they are never a second source of truth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub artifact_version: u32,
    pub method: MethodContract,
    pub architecture: String,
    pub source_fingerprint: String,
    pub plan_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_contract: Option<RuntimeContract>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<TransformPlan>,
    pub output_layout: OutputLayout,
    pub config_blake3: String,
    pub sidecar_blake3: BTreeMap<String, String>,
    pub tensors: Vec<TensorManifest>,
    pub standard_hf_checkpoint: bool,
    pub secret_key_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_id: Option<String>,
}

impl Manifest {
    pub fn from_plan(
        plan: &TransformPlan,
        config_bytes: &[u8],
        sidecar_blake3: BTreeMap<String, String>,
        tensors: Vec<TensorManifest>,
    ) -> Self {
        Self {
            artifact_version: plan.version,
            method: plan.method.clone(),
            architecture: plan.architecture.clone(),
            source_fingerprint: plan.source_fingerprint.to_string(),
            plan_hash: plan.plan_hash.to_string(),
            layout_hash: layout_hash(&plan.output_layout).ok(),
            runtime_contract: Some(plan.runtime_contract.clone()),
            plan: Some(plan.clone()),
            output_layout: plan.output_layout.clone(),
            config_blake3: blake3::hash(config_bytes).to_hex().to_string(),
            sidecar_blake3,
            tensors,
            standard_hf_checkpoint: plan.runtime_contract.standard_hf_checkpoint,
            secret_key_id: None,
            secret_id: plan.secret_id.clone(),
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
        match manifest.artifact_version {
            MANIFEST_VERSION | TOKEN_MANIFEST_VERSION => {
                manifest.validate_legacy(&path)?;
            }
            SCHEMA_VERSION | aloepri_core::keymat::KEYMAT_SCHEMA_VERSION => {
                manifest.validate_current(&path)?;
            }
            version => {
                return Err(CompilerError::UnsupportedVersion { version });
            }
        }
        Ok(manifest)
    }

    fn validate_legacy(&self, path: &Path) -> Result<()> {
        if self.runtime_contract.is_some() || self.plan.is_some() || self.layout_hash.is_some() {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: "a legacy manifest must not carry v3 contract fields".into(),
            });
        }
        let identity = self.method == MethodContract::identity();
        let token = self.method == MethodContract::aloepri_token();
        let token_secret_valid = self.secret_id.as_deref().is_some_and(valid_secret_id);
        if (!identity && !token)
            || (self.artifact_version == MANIFEST_VERSION
                && (!identity || self.secret_id.is_some()))
            || (self.artifact_version == TOKEN_MANIFEST_VERSION && (!token || !token_secret_valid))
            || !self.standard_hf_checkpoint
        {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: "manifest method, version, and secret metadata disagree".into(),
            });
        }
        Ok(())
    }

    fn validate_current(&self, path: &Path) -> Result<()> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: "a v3 manifest must embed its plan".into(),
            })?;
        let runtime =
            self.runtime_contract
                .as_ref()
                .ok_or_else(|| CompilerError::InvalidArtifact {
                    path: path.to_owned(),
                    reason: "a v3 manifest must carry a runtime contract".into(),
                })?;
        let layout_hash =
            self.layout_hash
                .as_ref()
                .ok_or_else(|| CompilerError::InvalidArtifact {
                    path: path.to_owned(),
                    reason: "a v3 manifest must carry a layout hash".into(),
                })?;
        plan.validate()
            .map_err(|error| CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("embedded plan is invalid: {error}"),
            })?;
        plan.verify_hash()
            .map_err(|error| CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("embedded plan hash is invalid: {error}"),
            })?;
        let consistent = plan.version == self.artifact_version
            && plan.method == self.method
            && plan.architecture == self.architecture
            && plan.source_fingerprint.to_string() == self.source_fingerprint
            && plan.plan_hash.to_string() == self.plan_hash
            && plan.output_layout == self.output_layout
            && plan.secret_id == self.secret_id
            && plan.runtime_contract == *runtime
            && self.standard_hf_checkpoint == runtime.standard_hf_checkpoint
            && layout_hash == &layout_hash_of(&self.output_layout)?
            && layout_hash == &layout_hash_of(&plan.output_layout)?;
        if !consistent {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: "manifest fields disagree with the embedded plan".into(),
            });
        }
        Ok(())
    }
}

/// BLAKE3 over the canonical JSON encoding of an output layout.
pub fn layout_hash(layout: &OutputLayout) -> Result<String> {
    layout_hash_of(layout)
}

fn layout_hash_of(layout: &OutputLayout) -> Result<String> {
    let bytes =
        serde_json::to_vec(layout).map_err(|error| CompilerError::Invariant(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn valid_secret_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}
