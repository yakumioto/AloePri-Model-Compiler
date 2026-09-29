use crate::{
    atomic::{OutputLock, publish_no_replace, sync_directory},
    checkpoint::Checkpoint,
    hf::HfArtifact,
    layout::plan_output_layout,
    manifest::{Manifest, TensorManifest},
    verify::verify_artifact,
    writer::StreamingWriter,
};
use aloepri_core::{
    backend::{ArtifactBackend, OutputLockGuard, VerificationOutcome},
    error::{CompilerError, Result, io_error},
    io::OutputWriter,
    model::ModelArtifact,
    plan::{OutputLayout, TransformConfig, TransformPlan},
    types::OperationId,
};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Default)]
pub struct HfBackend;

impl ArtifactBackend for HfBackend {
    fn open(&self, root: &Path) -> Result<Box<dyn ModelArtifact>> {
        Ok(Box::new(HfArtifact::open(root)?))
    }

    fn plan_output(
        &self,
        artifact: &dyn ModelArtifact,
        config: &TransformConfig,
    ) -> Result<OutputLayout> {
        plan_output_layout(artifact.tensors(), config.max_shard_size)
    }

    fn staging_has_output(&self, staging: &Path, layout: &OutputLayout) -> bool {
        let _ = layout;
        staging.join("model.safetensors.index.json").is_file()
            || fs::read_dir(staging).is_ok_and(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "safetensors")
                })
            })
    }

    fn create_writer(
        &self,
        staging: &Path,
        layout: &OutputLayout,
    ) -> Result<Box<dyn OutputWriter>> {
        Ok(Box::new(StreamingWriter::create(staging, layout.clone())?))
    }

    fn resume_writer(
        &self,
        staging: &Path,
        layout: &OutputLayout,
    ) -> Result<Box<dyn OutputWriter>> {
        Ok(Box::new(StreamingWriter::resume(staging, layout.clone())?))
    }

    fn prepare_staging(
        &self,
        staging: &Path,
        artifact: &dyn ModelArtifact,
        resume: bool,
    ) -> Result<()> {
        let config_path = staging.join("config.json");
        if config_path.exists() {
            let existing =
                fs::read(&config_path).map_err(|source| io_error(&config_path, source))?;
            if existing != artifact.config_bytes() {
                return Err(CompilerError::ResumeMismatch {
                    reason: "staging config differs from the source".into(),
                });
            }
        } else {
            fs::write(&config_path, artifact.config_bytes())
                .map_err(|source| io_error(&config_path, source))?;
        }
        for (name, bytes) in artifact.sidecars()? {
            let path = staging.join(&name);
            if path.exists() {
                if resume && fs::read(&path).map_err(|source| io_error(&path, source))? == bytes {
                    continue;
                }
                return Err(CompilerError::ResumeMismatch {
                    reason: format!("staging sidecar {name} differs from the source"),
                });
            }
            fs::write(&path, bytes).map_err(|source| io_error(&path, source))?;
        }
        Ok(())
    }

    fn load_checkpoint(
        &self,
        path: &Path,
        plan: &TransformPlan,
    ) -> Result<BTreeMap<OperationId, String>> {
        let checkpoint = Checkpoint::load(path)?;
        checkpoint.validate_against(plan)?;
        Ok(checkpoint.completed)
    }

    fn store_checkpoint(
        &self,
        path: &Path,
        plan: &TransformPlan,
        completed: &BTreeMap<OperationId, String>,
    ) -> Result<()> {
        Checkpoint::with_completed(plan, completed)?.store(path)
    }

    fn finalize(
        &self,
        staging: &Path,
        plan: &TransformPlan,
        artifact: &dyn ModelArtifact,
        writer: &mut dyn OutputWriter,
    ) -> Result<()> {
        let mut tensors = Vec::new();
        for descriptor in artifact.tensors() {
            let digest = writer.hash_tensor(&descriptor.name)?;
            tensors.push(TensorManifest {
                name: descriptor.name.clone(),
                shape: descriptor.shape.clone(),
                dtype: descriptor.dtype,
                byte_length: descriptor.byte_length.0,
                blake3: digest.to_string(),
            });
        }
        let sidecar_hashes = artifact
            .sidecars()?
            .iter()
            .map(|(name, bytes)| (name.clone(), blake3::hash(bytes).to_hex().to_string()))
            .collect();
        let manifest = Manifest::from_plan(plan, artifact.config_bytes(), sidecar_hashes, tensors);
        manifest.write(staging)?;
        sync_directory(staging)
    }

    fn output_matches_plan(&self, output: &Path, plan: &TransformPlan) -> bool {
        Manifest::read(output).is_ok_and(|manifest| {
            manifest.artifact_version == plan.version
                && manifest.method == plan.method
                && manifest.secret_id == plan.secret_id
                && manifest.source_fingerprint == plan.source_fingerprint.to_string()
                && manifest.plan_hash == plan.plan_hash.to_string()
        })
    }

    fn publish(&self, staging: &Path, output: &Path) -> Result<()> {
        publish_no_replace(staging, output)
    }

    fn lock_output(&self, path: &Path) -> Result<Box<dyn OutputLockGuard>> {
        Ok(Box::new(OutputLock::acquire(path)?))
    }

    fn verify(&self, root: &Path) -> Result<VerificationOutcome> {
        verify_artifact(root)
    }
}
