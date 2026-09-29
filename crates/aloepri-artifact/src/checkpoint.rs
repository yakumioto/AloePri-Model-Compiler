use crate::manifest::layout_hash;
use aloepri_core::{
    error::{CompilerError, Result, io_error, json_error},
    plan::{MethodContract, RuntimeContract, SCHEMA_VERSION, TransformPlan},
    types::OperationId,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

/// Legacy checkpoint schema written alongside v1 identity artifacts.
pub const CHECKPOINT_VERSION: u32 = 1;
/// Legacy checkpoint schema written alongside v2 token artifacts.
pub const TOKEN_CHECKPOINT_VERSION: u32 = 2;

/// Resumable execution state.
///
/// Only the current schema may be resumed: a legacy v1/v2 checkpoint belongs to
/// a legacy execution model and is rejected rather than silently upgraded. The
/// runtime contract is part of the binding so a different physical expansion,
/// method or source fingerprint cannot reuse a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub schema_version: u32,
    pub source_fingerprint: String,
    pub plan_hash: String,
    pub output_layout_hash: String,
    pub method: MethodContract,
    pub secret_key_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_contract: Option<RuntimeContract>,
    pub completed: BTreeMap<OperationId, String>,
}

impl Checkpoint {
    pub fn new(plan: &TransformPlan) -> Result<Self> {
        Self::with_completed(plan, &BTreeMap::new())
    }

    pub fn with_completed(
        plan: &TransformPlan,
        completed: &BTreeMap<OperationId, String>,
    ) -> Result<Self> {
        Ok(Self {
            schema_version: plan.version,
            source_fingerprint: plan.source_fingerprint.to_string(),
            plan_hash: plan.plan_hash.to_string(),
            output_layout_hash: layout_hash(&plan.output_layout)?,
            method: plan.method.clone(),
            secret_key_id: None,
            secret_id: plan.secret_id.clone(),
            runtime_contract: Some(plan.runtime_contract.clone()),
            completed: completed.clone(),
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|source| io_error(path, source))?;
        crate::header::reject_duplicate_keys(&bytes).map_err(|reason| {
            CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("invalid or duplicate checkpoint JSON: {reason}"),
            }
        })?;
        let checkpoint: Self =
            serde_json::from_slice(&bytes).map_err(|source| json_error(path, source))?;
        if checkpoint.schema_version != SCHEMA_VERSION {
            return Err(CompilerError::UnsupportedVersion {
                version: checkpoint.schema_version,
            });
        }
        if checkpoint.runtime_contract.is_none() {
            return Err(CompilerError::ResumeMismatch {
                reason: "a resumable checkpoint must carry a runtime contract".into(),
            });
        }
        Ok(checkpoint)
    }

    pub fn store(&self, path: &Path) -> Result<()> {
        let temporary = temporary_path(path);
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| CompilerError::Invariant(error.to_string()))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| io_error(&temporary, source))?;
        file.write_all(&bytes)
            .map_err(|source| io_error(&temporary, source))?;
        file.sync_all()
            .map_err(|source| io_error(&temporary, source))?;
        fs::rename(&temporary, path).map_err(|source| io_error(path, source))?;
        if let Some(parent) = path.parent() {
            let directory = File::open(parent).map_err(|source| io_error(parent, source))?;
            directory
                .sync_all()
                .map_err(|source| io_error(parent, source))?;
        }
        Ok(())
    }

    pub fn validate_against(&self, plan: &TransformPlan) -> Result<()> {
        if self.schema_version != plan.version
            || self.source_fingerprint != plan.source_fingerprint.to_string()
            || self.plan_hash != plan.plan_hash.to_string()
            || self.output_layout_hash != layout_hash(&plan.output_layout)?
            || self.method != plan.method
            || self.secret_key_id.is_some()
            || self.secret_id != plan.secret_id
            || self.runtime_contract.as_ref() != Some(&plan.runtime_contract)
        {
            return Err(CompilerError::ResumeMismatch {
                reason: "checkpoint contract does not match the current plan".into(),
            });
        }
        Ok(())
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.to_owned();
    let suffix = format!(".tmp-{}", std::process::id());
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("checkpoint");
    temporary.set_file_name(format!("{name}{suffix}"));
    temporary
}
