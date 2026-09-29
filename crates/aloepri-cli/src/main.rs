mod args;

use aloepri_architecture::Registry;
use aloepri_artifact::{HfArtifact, HfBackend};
use aloepri_core::{
    Compiler, ModelArtifact, TransformRequest,
    plan::{MethodContract, SecretBinding, TransformConfig},
    types::ByteLength,
};
use aloepri_secret::ClientSecret;
use aloepri_transform::{IdentityExecutor, TokenPermutationExecutor};
use anyhow::{Context, Result, anyhow};
use args::{Cli, Command};
use clap::Parser;
use std::{
    fs,
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("info")
        .init();
    match Cli::parse().command {
        Command::Inspect { model } => inspect(&model),
        Command::Plan(args) => {
            let method = args::resolve_method(args.identity, args.method.as_deref())?;
            if method != MethodContract::identity() {
                return Err(anyhow!(
                    "aloepri-token planning requires an existing client secret; use transform"
                ));
            }
            let config = transform_config(args.memory_limit, args.max_shard_size, method, None)?;
            let compiler = identity_compiler();
            let plan = compiler
                .plan(&args.model, &config)
                .context("build identity plan")?;
            println!("{}", serde_json::to_string_pretty(&plan_summary(&plan))?);
            Ok(())
        }
        Command::Transform(args) => transform(args),
        Command::Verify { model } => {
            let report = identity_compiler()
                .verify(&model)
                .context("verify model artifact")?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn transform(args: args::TransformArgs) -> Result<()> {
    let method = args::resolve_method(args.identity, args.method.as_deref())?;
    match method {
        method if method == MethodContract::identity() => {
            if args.secret_output.is_some() {
                return Err(anyhow!("--secret-output is only valid for aloepri-token"));
            }
            let config = transform_config(args.memory_limit, args.max_shard_size, method, None)?;
            let request = TransformRequest {
                input: args.model,
                output: args.output,
                config,
                resume: args.resume,
            };
            let report = identity_compiler()
                .transform(&request)
                .context("run identity transform")?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        method if method == MethodContract::aloepri_token() => transform_token(args, method),
        _ => unreachable!("method parser only returns supported contracts"),
    }
}

fn transform_token(args: args::TransformArgs, method: MethodContract) -> Result<()> {
    let secret_path = args
        .secret_output
        .as_deref()
        .ok_or_else(|| anyhow!("aloepri-token requires --secret-output"))?;
    validate_secret_path(&args.model, &args.output, secret_path, args.resume)?;
    if !args.resume && (args.output.exists() || staging_path(&args.output).exists()) {
        return Err(anyhow!(
            "fresh token transform cannot reuse an existing output or staging directory"
        ));
    }

    let artifact = HfArtifact::open(&args.model).context("open input model")?;
    let source_fingerprint = artifact.fingerprint().context("fingerprint input model")?;
    let vocab_size = artifact
        .config()
        .get("vocab_size")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("input model config is missing vocab_size"))?;
    let secret = if args.resume {
        ClientSecret::read(secret_path).context("read client secret")?
    } else {
        if secret_path.exists() {
            return Err(anyhow!("client secret path already exists"));
        }
        let secret = ClientSecret::generate(source_fingerprint, vocab_size)
            .context("generate client secret")?;
        secret
            .write_new(secret_path)
            .context("persist client secret")?;
        secret
    };
    secret
        .validate_for_model(source_fingerprint, vocab_size)
        .context("validate client secret against input model")?;
    let binding = secret.binding().context("read client secret binding")?;
    let config = transform_config(
        args.memory_limit,
        args.max_shard_size,
        method,
        Some(binding.clone()),
    )?;
    let executor = TokenPermutationExecutor::new(secret.inverse_permutation().to_vec(), binding)?;
    let request = TransformRequest {
        input: args.model,
        output: args.output,
        config,
        resume: args.resume,
    };
    let report = Compiler::new(HfBackend, Registry::new(), executor)
        .transform(&request)
        .context("run aloepri-token transform")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn identity_compiler() -> Compiler<HfBackend, Registry, IdentityExecutor> {
    Compiler::new(HfBackend, Registry::new(), IdentityExecutor::default())
}

fn transform_config(
    memory_limit: u64,
    max_shard_size: u64,
    method: MethodContract,
    secret_binding: Option<SecretBinding>,
) -> Result<TransformConfig> {
    if method == MethodContract::identity() && secret_binding.is_some() {
        return Err(anyhow!("identity cannot carry a client secret binding"));
    }
    if method == MethodContract::aloepri_token() && secret_binding.is_none() {
        return Err(anyhow!("aloepri-token requires a client secret binding"));
    }
    let config = TransformConfig {
        method,
        output_dtype: aloepri_core::types::OutputDType::Preserve,
        memory_limit: ByteLength(memory_limit),
        max_shard_size: ByteLength(max_shard_size),
        workers: 1,
        secret_binding,
    };
    aloepri_secret::validate_v0_1_boundary(&config)?;
    Ok(config)
}

fn inspect(model: &Path) -> Result<()> {
    let report = identity_compiler()
        .inspect(model)
        .context("inspect model artifact")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn plan_summary(plan: &aloepri_core::TransformPlan) -> serde_json::Value {
    serde_json::json!({
        "version": plan.version,
        "method": plan.method,
        "secret_id": plan.secret_id,
        "architecture": plan.architecture,
        "runtime_contract": plan.runtime_contract,
        "source_fingerprint": plan.source_fingerprint.to_string(),
        "plan_hash": plan.plan_hash.to_string(),
        "source_tensor_count": plan.source_inventory.len(),
        "output_tensor_count": plan.output_inventory.len(),
        "operations": plan.operations.len(),
        "source_payload_bytes": plan.source_inventory.iter().map(|tensor| tensor.byte_length.0).sum::<u64>(),
        "output_payload_bytes": plan.output_inventory.iter().map(|tensor| tensor.byte_length.0).sum::<u64>(),
        "shards": plan.output_layout.shards.iter().map(|shard| serde_json::json!({
            "id": shard.id,
            "filename": shard.filename,
            "payload_bytes": shard.payload_length.0,
            "file_bytes": shard.file_length.0,
        })).collect::<Vec<_>>(),
        "memory_estimate": plan.memory_estimate,
    })
}

fn staging_path(output: &Path) -> PathBuf {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output.file_name().unwrap_or_default().to_string_lossy();
    parent.join(format!(".{name}.aloepri-work"))
}

fn validate_secret_path(input: &Path, output: &Path, secret: &Path, resume: bool) -> Result<()> {
    let input = input
        .canonicalize()
        .with_context(|| format!("canonicalize input {}", input.display()))?;
    let output = canonical_candidate(output)?;
    if output == input || output.starts_with(&input) || input.starts_with(&output) {
        return Err(anyhow!(
            "input and output paths must not be equal or nested"
        ));
    }
    let secret_existing = if secret.exists() {
        let metadata = fs::symlink_metadata(secret)
            .with_context(|| format!("inspect secret path {}", secret.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(anyhow!("secret path must not be a symlink"));
        }
        Some(
            secret
                .canonicalize()
                .with_context(|| format!("canonicalize secret {}", secret.display()))?,
        )
    } else {
        let parent = secret
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        Some(
            parent
                .canonicalize()
                .with_context(|| format!("canonicalize secret parent {}", parent.display()))?
                .join(secret.file_name().unwrap_or_default()),
        )
    };
    let secret = secret_existing.expect("secret path candidate is always present");
    let work = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".{}.aloepri-work",
            output.file_name().unwrap_or_default().to_string_lossy()
        ));
    if secret == input
        || secret.starts_with(&input)
        || secret == output
        || secret.starts_with(&output)
        || secret == work
        || secret.starts_with(&work)
    {
        return Err(anyhow!(
            "client secret must be outside input, output, and compiler staging paths"
        ));
    }
    if !resume && secret.exists() {
        return Err(anyhow!("client secret path already exists"));
    }
    Ok(())
}

fn canonical_candidate(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return path
            .canonicalize()
            .with_context(|| format!("canonicalize path {}", path.display()));
    }
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent
        .canonicalize()
        .with_context(|| format!("canonicalize parent {}", parent.display()))?
        .join(path.file_name().unwrap_or_default()))
}
