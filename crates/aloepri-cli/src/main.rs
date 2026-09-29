mod args;

use aloepri_architecture::Registry;
use aloepri_artifact::{
    atomic::{OutputLock, publish_no_replace, sync_directory},
    checkpoint::Checkpoint,
    hf::HfArtifact,
    layout::plan_output_layout,
    manifest::{Manifest, TensorManifest},
    verify::verify_artifact,
    writer::StreamingWriter,
};
use aloepri_core::{
    Compiler, MemoryBudget,
    error::{CompilerError, Result as CoreResult, io_error},
    io::TensorWriter,
    model::ModelArtifact,
    plan::{TransformConfig, TransformPlan},
    types::ByteLength,
};
use aloepri_secret::validate_v0_1_boundary;
use aloepri_transform::IdentityExecutor;
use anyhow::{Context, Result, anyhow};
use args::{Cli, Command};
use clap::Parser;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("info")
        .init();
    match Cli::parse().command {
        Command::Inspect { model } => inspect(&model),
        Command::Plan(args) => {
            args::require_identity(args.identity)?;
            let config = transform_config(args.memory_limit, args.max_shard_size);
            let artifact = HfArtifact::open(&args.model).context("open model artifact")?;
            validate_v0_1_boundary(&config)?;
            let plan = make_plan(&artifact, &config)?;
            println!("{}", serde_json::to_string_pretty(&plan_summary(&plan))?);
            Ok(())
        }
        Command::Transform(args) => {
            args::require_identity(args.identity)?;
            let config = transform_config(args.memory_limit, args.max_shard_size);
            transform(&args.model, &args.output, &config, args.resume)
        }
        Command::Verify { model } => {
            let report = verify_artifact(&model).context("verify model artifact")?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn inspect(model: &Path) -> Result<()> {
    let artifact = HfArtifact::open(model).context("open model artifact")?;
    let fingerprint = artifact
        .fingerprint()
        .context("fingerprint model artifact")?;
    let report = json!({
        "root": artifact.root(),
        "architecture": artifact.model_spec().architecture,
        "tensor_count": artifact.tensors().len(),
        "payload_bytes": artifact.total_payload_bytes(),
        "shards": artifact.shards(),
        "aliases": artifact.model_spec().aliases,
        "fingerprint": fingerprint.to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn transform_config(memory_limit: u64, max_shard_size: u64) -> TransformConfig {
    TransformConfig {
        memory_limit: ByteLength(memory_limit),
        max_shard_size: ByteLength(max_shard_size),
        ..TransformConfig::default()
    }
}

fn make_plan(artifact: &HfArtifact, config: &TransformConfig) -> CoreResult<TransformPlan> {
    let compiler = Compiler::new(Registry::new());
    let draft = compiler.plan_draft(artifact, config)?;
    let fingerprint = artifact.fingerprint()?;
    let layout = plan_output_layout(artifact.tensors(), config.max_shard_size)?;
    TransformPlan::from_draft(
        draft,
        fingerprint,
        artifact.tensors().to_vec(),
        layout,
        config,
    )
}

fn plan_summary(plan: &TransformPlan) -> serde_json::Value {
    json!({
        "version": plan.version,
        "method": plan.method,
        "architecture": plan.architecture,
        "source_fingerprint": plan.source_fingerprint.to_string(),
        "plan_hash": plan.plan_hash.to_string(),
        "tensor_count": plan.source_inventory.len(),
        "operations": plan.operations.len(),
        "shards": plan.output_layout.shards.iter().map(|shard| json!({
            "id": shard.id,
            "filename": shard.filename,
            "payload_bytes": shard.payload_length.0,
            "file_bytes": shard.file_length.0,
        })).collect::<Vec<_>>(),
        "memory_estimate": plan.memory_estimate,
    })
}

fn transform(input: &Path, output: &Path, config: &TransformConfig, resume: bool) -> Result<()> {
    let artifact = HfArtifact::open(input).context("open input artifact")?;
    validate_output_path(artifact.root(), output)?;
    let plan = make_plan(&artifact, config).context("build transform plan")?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
    let output_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("output must have a valid file name"))?;
    let work = parent.join(format!(".{output_name}.aloepri-work"));
    if output.exists() {
        if resume
            && output.is_dir()
            && let Ok(manifest) = Manifest::read(output)
            && manifest.source_fingerprint == plan.source_fingerprint.to_string()
            && manifest.plan_hash == plan.plan_hash.to_string()
        {
            let report = verify_artifact(output).context("verify already published output")?;
            if report.manifest_verified {
                if work.exists() {
                    fs::remove_dir_all(&work).map_err(|source| io_error(&work, source))?;
                }
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }
        }
        return Err(anyhow!("output already exists: {}", output.display()));
    }
    if work.exists() && !resume {
        return Err(anyhow!(
            "existing staging directory requires --resume: {}",
            work.display()
        ));
    }
    if !work.exists() {
        fs::create_dir_all(&work).map_err(|source| io_error(&work, source))?;
    }
    let candidate = work.join("artifact");
    if !candidate.exists() {
        fs::create_dir_all(&candidate).map_err(|source| io_error(&candidate, source))?;
    }
    let lock_path = parent.join(format!(".{output_name}.aloepri.lock"));
    let _lock = OutputLock::acquire(&lock_path).context("acquire output lock")?;
    copy_sidecars(&artifact, &candidate, resume)?;

    let mut writer = if resume && candidate.join("model.safetensors").exists()
        || resume && candidate.join("model.safetensors.index.json").exists()
    {
        StreamingWriter::resume(&candidate, plan.output_layout.clone())
            .context("resume output writer")?
    } else {
        StreamingWriter::create(&candidate, plan.output_layout.clone())
            .context("create output writer")?
    };
    let checkpoint_path = work.join("checkpoint.json");
    let mut checkpoint = if resume {
        let checkpoint = Checkpoint::load(&checkpoint_path).context("load checkpoint")?;
        checkpoint
            .validate_against(&plan)
            .context("validate checkpoint")?;
        checkpoint
    } else {
        let checkpoint = Checkpoint::new(&plan).context("create checkpoint")?;
        checkpoint
            .store(&checkpoint_path)
            .context("persist initial checkpoint")?;
        checkpoint
    };
    validate_completed(&checkpoint, &plan, &mut writer)?;
    let budget = MemoryBudget::new(config.memory_limit.0);
    let identity = IdentityExecutor::default();
    for operation in &plan.operations {
        if checkpoint.completed.contains_key(&operation.id) {
            continue;
        }
        let digest = identity
            .execute_operation(&artifact, &plan, operation, &mut writer, &budget)
            .context("execute identity operation")?;
        writer.sync().context("sync completed operation")?;
        checkpoint
            .completed
            .insert(operation.id, digest.to_string());
        checkpoint
            .store(&checkpoint_path)
            .context("persist checkpoint")?;
    }
    writer.write_index().context("write output index")?;
    writer.sync().context("sync output shards")?;

    let mut tensor_manifests = Vec::new();
    for descriptor in artifact.tensors() {
        let digest = writer
            .hash_tensor(&descriptor.name)
            .context("hash output tensor")?;
        tensor_manifests.push(TensorManifest {
            name: descriptor.name.clone(),
            shape: descriptor.shape.clone(),
            dtype: descriptor.dtype,
            byte_length: descriptor.byte_length.0,
            blake3: digest.to_string(),
        });
    }
    let sidecar_hashes = sidecar_hashes(&artifact.sidecar_bytes()?)?;
    let manifest = Manifest::from_plan(
        &plan,
        artifact.config_bytes(),
        sidecar_hashes,
        tensor_manifests,
    );
    manifest
        .write(&candidate)
        .context("write output manifest")?;
    fs::write(candidate.join("config.json"), artifact.config_bytes())
        .map_err(|source| io_error(candidate.join("config.json"), source))?;
    sync_directory(&candidate).context("sync candidate artifact")?;
    let report = verify_artifact(&candidate).context("verify candidate artifact")?;
    publish_no_replace(&candidate, output).context("publish output atomically")?;
    sync_directory(&work).ok();
    fs::remove_dir_all(&work).map_err(|source| io_error(&work, source))?;
    let _ = fs::remove_file(&lock_path);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
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
        return Err(anyhow!(
            "input and output paths must not be equal or nested"
        ));
    }
    Ok(())
}

fn copy_sidecars(artifact: &HfArtifact, destination: &Path, resume: bool) -> CoreResult<()> {
    let config_path = destination.join("config.json");
    if config_path.exists() {
        let existing = fs::read(&config_path).map_err(|source| io_error(&config_path, source))?;
        if existing != artifact.config_bytes() {
            return Err(CompilerError::ResumeMismatch {
                reason: "staging config differs from source".into(),
            });
        }
    } else {
        fs::write(&config_path, artifact.config_bytes())
            .map_err(|source| io_error(&config_path, source))?;
    }
    for (name, bytes) in artifact.sidecar_bytes()? {
        let path = destination.join(&name);
        if path.exists() {
            if resume && fs::read(&path).map_err(|source| io_error(&path, source))? == bytes {
                continue;
            }
            return Err(CompilerError::ResumeMismatch {
                reason: format!("staging sidecar {name} differs from source"),
            });
        }
        fs::write(&path, bytes).map_err(|source| io_error(&path, source))?;
    }
    Ok(())
}

fn sidecar_hashes(sidecars: &BTreeMap<String, Vec<u8>>) -> CoreResult<BTreeMap<String, String>> {
    Ok(sidecars
        .iter()
        .map(|(name, bytes)| (name.clone(), blake3::hash(bytes).to_hex().to_string()))
        .collect())
}

fn validate_completed(
    checkpoint: &Checkpoint,
    plan: &TransformPlan,
    writer: &mut StreamingWriter,
) -> CoreResult<()> {
    let mut expected = BTreeSet::new();
    let mut gap = false;
    for operation in &plan.operations {
        if let Some(digest) = checkpoint.completed.get(&operation.id) {
            if gap || !expected.insert(operation.id) {
                return Err(CompilerError::ResumeMismatch {
                    reason: "completed operations are not a contiguous prefix".into(),
                });
            }
            let actual = writer.hash_tensor(&operation.tensor)?.to_string();
            if actual != *digest {
                return Err(CompilerError::ResumeMismatch {
                    reason: format!(
                        "completed tensor {} does not match checkpoint",
                        operation.tensor
                    ),
                });
            }
        } else {
            gap = true;
        }
    }
    if checkpoint.completed.len() != expected.len() {
        return Err(CompilerError::ResumeMismatch {
            reason: "checkpoint references an unknown operation".into(),
        });
    }
    Ok(())
}
