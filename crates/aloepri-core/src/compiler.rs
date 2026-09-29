use crate::{
    backend::{
        ArtifactBackend, InspectionReport, ShardSummary, TransformReport, TransformRequest,
        VerificationOutcome,
    },
    error::{CompilerError, Result, io_error},
    executor::TransformExecutor,
    memory::MemoryBudget,
    model::{ArchitectureRegistry, ModelArtifact},
    plan::{TransformConfig, TransformPlan},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

pub struct Compiler<B, R, E> {
    backend: B,
    registry: R,
    executor: E,
}

impl<B: ArtifactBackend, R: ArchitectureRegistry, E: TransformExecutor> Compiler<B, R, E> {
    pub fn new(backend: B, registry: R, executor: E) -> Self {
        Self {
            backend,
            registry,
            executor,
        }
    }

    pub fn inspect(&self, root: &Path) -> Result<InspectionReport> {
        let artifact = self.backend.open(root)?;
        let fingerprint = artifact.fingerprint()?;
        let shards: Vec<ShardSummary> = artifact.shards();
        Ok(InspectionReport {
            architecture: artifact.model_spec().architecture.clone(),
            tensor_count: artifact.tensors().len(),
            payload_bytes: artifact
                .tensors()
                .iter()
                .map(|tensor| tensor.byte_length.0)
                .sum(),
            shards,
            aliases: artifact.model_spec().aliases.clone(),
            fingerprint: fingerprint.to_string(),
        })
    }

    pub fn plan(&self, root: &Path, config: &TransformConfig) -> Result<TransformPlan> {
        let artifact = self.backend.open(root)?;
        self.build_plan(artifact.as_ref(), config)
    }

    pub fn transform(&self, request: &TransformRequest) -> Result<TransformReport> {
        let artifact = self.backend.open(&request.input)?;
        validate_output_path(artifact.root(), &request.output)?;
        let plan = self.build_plan(artifact.as_ref(), &request.config)?;
        self.executor.requirements(&request.config)?;

        let output = request.output.as_path();
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
        let output_name = output
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| CompilerError::Invariant("output has no usable file name".into()))?;
        let work = parent.join(format!(".{output_name}.aloepri-work"));
        let candidate = work.join("artifact");

        if output.exists() {
            if request.resume
                && output.is_dir()
                && self.backend.output_matches_plan(output, &plan)
                && let Ok(report) = self.backend.verify(output)
                && report.manifest_verified
            {
                if work.exists() {
                    fs::remove_dir_all(&work).map_err(|source| io_error(&work, source))?;
                }
                return Ok(self.report(&plan, plan.operations.len(), report));
            }
            return Err(CompilerError::AlreadyExists {
                path: output.to_owned(),
            });
        }
        if work.exists() && !request.resume {
            return Err(CompilerError::ResumeMismatch {
                reason: format!(
                    "staging directory {} already exists; re-run with resume enabled",
                    work.display()
                ),
            });
        }
        fs::create_dir_all(&candidate).map_err(|source| io_error(&candidate, source))?;
        let lock_path = parent.join(format!(".{output_name}.aloepri.lock"));
        let _lock = self.backend.lock_output(&lock_path)?;
        self.backend
            .prepare_staging(&candidate, artifact.as_ref(), request.resume)?;

        // Interruption leaves the planned shard files in place even though the
        // index is written only after every operation, so resume is decided by
        // the output layout rather than by the single-file name.
        let mut writer = if request.resume
            && self
                .backend
                .staging_has_output(&candidate, &plan.output_layout)
        {
            self.backend
                .resume_writer(&candidate, &plan.output_layout)?
        } else {
            self.backend
                .create_writer(&candidate, &plan.output_layout)?
        };

        let checkpoint_path = work.join("checkpoint.json");
        let mut completed = if request.resume {
            let completed = self.backend.load_checkpoint(&checkpoint_path, &plan)?;
            self.validate_completed(&plan, &completed, writer.as_mut())?;
            completed
        } else {
            let completed = BTreeMap::new();
            self.backend
                .store_checkpoint(&checkpoint_path, &plan, &completed)?;
            completed
        };

        let budget = MemoryBudget::new(request.config.memory_limit.0);
        for operation in &plan.operations {
            if completed.contains_key(&operation.id) {
                continue;
            }
            let digest = self.executor.execute_operation(
                artifact.as_ref(),
                operation,
                writer.as_mut(),
                &budget,
            )?;
            writer.sync()?;
            completed.insert(operation.id, digest.to_string());
            self.backend
                .store_checkpoint(&checkpoint_path, &plan, &completed)?;
        }
        writer.write_index()?;
        writer.sync()?;
        self.backend
            .finalize(&candidate, &plan, artifact.as_ref(), writer.as_mut())?;
        let report = self.backend.verify(&candidate)?;
        self.backend.publish(&candidate, output)?;
        if work.exists() {
            fs::remove_dir_all(&work).map_err(|source| io_error(&work, source))?;
        }
        Ok(self.report(&plan, completed.len(), report))
    }

    pub fn verify(&self, root: &Path) -> Result<VerificationOutcome> {
        self.backend.verify(root)
    }

    fn build_plan(
        &self,
        artifact: &dyn ModelArtifact,
        config: &TransformConfig,
    ) -> Result<TransformPlan> {
        let adapter = self.registry.detect(artifact)?;
        let draft = adapter.build_plan(artifact, config)?;
        let fingerprint = artifact.fingerprint()?;
        let layout = self.backend.plan_output(artifact, config)?;
        TransformPlan::from_draft(
            draft,
            fingerprint,
            artifact.tensors().to_vec(),
            layout,
            config,
        )
    }

    fn report(
        &self,
        plan: &TransformPlan,
        completed: usize,
        verification: VerificationOutcome,
    ) -> TransformReport {
        TransformReport {
            plan_hash: plan.plan_hash.to_string(),
            tensor_count: plan.operations.len(),
            payload_bytes: plan
                .source_inventory
                .iter()
                .map(|tensor| tensor.byte_length.0)
                .sum(),
            shards: plan
                .output_layout
                .shards
                .iter()
                .map(|shard| ShardSummary {
                    filename: shard.filename.clone(),
                    file_length: shard.file_length.0,
                    payload_length: shard.payload_length.0,
                })
                .collect(),
            completed_operations: completed,
            verification,
        }
    }

    fn validate_completed(
        &self,
        plan: &TransformPlan,
        completed: &BTreeMap<crate::types::OperationId, String>,
        writer: &mut dyn crate::io::OutputWriter,
    ) -> Result<()> {
        let mut expected = BTreeSet::new();
        let mut gap = false;
        for operation in &plan.operations {
            if let Some(digest) = completed.get(&operation.id) {
                if gap || !expected.insert(operation.id) {
                    return Err(CompilerError::ResumeMismatch {
                        reason: "completed operations are not a contiguous prefix".into(),
                    });
                }
                let actual = writer.hash_tensor(&operation.tensor)?.to_string();
                if actual != *digest {
                    return Err(CompilerError::ResumeMismatch {
                        reason: format!(
                            "completed tensor {} does not match the checkpoint",
                            operation.tensor
                        ),
                    });
                }
            } else {
                gap = true;
            }
        }
        if completed.len() != expected.len() {
            return Err(CompilerError::ResumeMismatch {
                reason: "checkpoint references an unknown operation".into(),
            });
        }
        Ok(())
    }
}

fn validate_output_path(input: &Path, output: &Path) -> Result<()> {
    let input = input
        .canonicalize()
        .map_err(|source| io_error(input, source))?;
    let output = if output.exists() {
        output
            .canonicalize()
            .map_err(|source| io_error(output, source))?
    } else {
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        parent
            .canonicalize()
            .map(|parent| parent.join(output.file_name().unwrap_or_default()))
            .map_err(|source| io_error(parent, source))?
    };
    if output == input || output.starts_with(&input) || input.starts_with(&output) {
        return Err(CompilerError::InvalidPlan {
            reason: "input and output paths must not be equal or nested".into(),
        });
    }
    Ok(())
}
