use crate::{
    error::Result,
    io::OutputWriter,
    model::ModelArtifact,
    plan::{OutputLayout, RuntimeContract, TransformConfig, TransformPlan},
    types::{OperationId, OutputTensorDescriptor},
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// Writer lifetime handle for a staging artifact. Dropping it releases the lock.
pub trait OutputLockGuard: Send {}

#[derive(Clone, Debug, Serialize)]
pub struct ShardSummary {
    pub filename: String,
    pub file_length: u64,
    pub payload_length: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct InspectionReport {
    pub architecture: String,
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub shards: Vec<ShardSummary>,
    pub aliases: BTreeMap<String, String>,
    pub fingerprint: String,
}

/// Layered verification result.
///
/// `structure_valid` proves the container parses; `manifest_verified` proves a
/// v3 contract matched its plan. `standard_hf_checkpoint` is a contract
/// declaration, not the result of loading the artifact with Transformers, and
/// `semantic_verification` records that no model-level check ran.
#[derive(Clone, Debug, Serialize)]
pub struct VerificationOutcome {
    pub structure_valid: bool,
    pub manifest_verified: bool,
    pub artifact_valid: bool,
    pub plan_verified: bool,
    pub standard_hf_checkpoint: Option<bool>,
    pub runtime_required: bool,
    pub verification_scope: String,
    pub semantic_verification: String,
    pub runtime_contract: Option<RuntimeContract>,
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub artifact_fingerprint: String,
}

#[derive(Clone, Debug)]
pub struct TransformRequest {
    pub input: PathBuf,
    pub output: PathBuf,
    pub config: TransformConfig,
    pub resume: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TransformReport {
    pub plan_hash: String,
    pub tensor_count: usize,
    pub source_tensor_count: usize,
    pub payload_bytes: u64,
    pub source_payload_bytes: u64,
    pub shards: Vec<ShardSummary>,
    pub completed_operations: usize,
    pub verification: VerificationOutcome,
}

/// Everything the compiler façade needs from an artifact format.
///
/// `aloepri-core` performs the ordering, staging, checkpoint and publication
/// logic; this contract carries the format-specific work (HF discovery, output
/// layout, safetensors writer, manifest/checkpoint persistence, atomic publish
/// and independent verification) so the core never depends on an artifact crate.
pub trait ArtifactBackend {
    fn open(&self, root: &Path) -> Result<Box<dyn ModelArtifact>>;

    /// Plan the physical layout from the output descriptors alone.
    fn plan_output(
        &self,
        outputs: &[OutputTensorDescriptor],
        config: &TransformConfig,
    ) -> Result<OutputLayout>;

    /// Reject a layout whose header/index/ranges are not internally consistent.
    /// Called before the writer creates any file.
    fn validate_output_layout(&self, layout: &OutputLayout) -> Result<()>;

    /// True when the staging directory already holds safetensors output, which
    /// is the state a mid-execution interruption leaves behind. Such staging is
    /// resumed and strictly validated rather than re-created.
    fn staging_has_output(&self, staging: &Path, layout: &OutputLayout) -> bool;

    fn create_writer(&self, staging: &Path, layout: &OutputLayout)
    -> Result<Box<dyn OutputWriter>>;

    fn resume_writer(&self, staging: &Path, layout: &OutputLayout)
    -> Result<Box<dyn OutputWriter>>;

    /// Copy the preserved config and allowlisted sidecars into staging.
    fn prepare_staging(
        &self,
        staging: &Path,
        artifact: &dyn ModelArtifact,
        resume: bool,
    ) -> Result<()>;

    /// Load and validate the checkpoint contract against `plan`.
    fn load_checkpoint(
        &self,
        path: &Path,
        plan: &TransformPlan,
    ) -> Result<BTreeMap<OperationId, String>>;

    fn store_checkpoint(
        &self,
        path: &Path,
        plan: &TransformPlan,
        completed: &BTreeMap<OperationId, String>,
    ) -> Result<()>;

    /// Write the manifest for a fully written staging artifact and sync it.
    fn finalize(
        &self,
        staging: &Path,
        plan: &TransformPlan,
        artifact: &dyn ModelArtifact,
        writer: &mut dyn OutputWriter,
    ) -> Result<()>;

    /// True when `output` is already published from an equivalent plan.
    fn output_matches_plan(&self, output: &Path, plan: &TransformPlan) -> bool;

    fn publish(&self, staging: &Path, output: &Path) -> Result<()>;

    fn lock_output(&self, path: &Path) -> Result<Box<dyn OutputLockGuard>>;

    fn verify(&self, root: &Path) -> Result<VerificationOutcome>;
}
