mod args;

use aloepri_architecture::Registry;
use aloepri_artifact::HfBackend;
use aloepri_core::{Compiler, TransformRequest, plan::TransformConfig, types::ByteLength};
use aloepri_secret::validate_v0_1_boundary;
use aloepri_transform::IdentityExecutor;
use anyhow::{Context, Result};
use args::{Cli, Command};
use clap::Parser;
use std::path::Path;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("info")
        .init();
    match Cli::parse().command {
        Command::Inspect { model } => inspect(&model),
        Command::Plan(args) => {
            args::require_identity(args.identity)?;
            let config = transform_config(args.memory_limit, args.max_shard_size)?;
            let compiler = compiler();
            let plan = compiler
                .plan(&args.model, &config)
                .context("build identity plan")?;
            println!("{}", serde_json::to_string_pretty(&plan_summary(&plan))?);
            Ok(())
        }
        Command::Transform(args) => {
            args::require_identity(args.identity)?;
            let config = transform_config(args.memory_limit, args.max_shard_size)?;
            let request = TransformRequest {
                input: args.model,
                output: args.output,
                config,
                resume: args.resume,
            };
            let report = compiler()
                .transform(&request)
                .context("run identity transform")?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Verify { model } => {
            let report = compiler().verify(&model).context("verify model artifact")?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn compiler() -> Compiler<HfBackend, Registry, IdentityExecutor> {
    Compiler::new(HfBackend, Registry::new(), IdentityExecutor::default())
}

fn inspect(model: &Path) -> Result<()> {
    let report = compiler()
        .inspect(model)
        .context("inspect model artifact")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn transform_config(memory_limit: u64, max_shard_size: u64) -> Result<TransformConfig> {
    let config = TransformConfig {
        memory_limit: ByteLength(memory_limit),
        max_shard_size: ByteLength(max_shard_size),
        ..TransformConfig::default()
    };
    validate_v0_1_boundary(&config)?;
    Ok(config)
}

fn plan_summary(plan: &aloepri_core::TransformPlan) -> serde_json::Value {
    serde_json::json!({
        "version": plan.version,
        "method": plan.method,
        "architecture": plan.architecture,
        "source_fingerprint": plan.source_fingerprint.to_string(),
        "plan_hash": plan.plan_hash.to_string(),
        "tensor_count": plan.source_inventory.len(),
        "operations": plan.operations.len(),
        "shards": plan.output_layout.shards.iter().map(|shard| serde_json::json!({
            "id": shard.id,
            "filename": shard.filename,
            "payload_bytes": shard.payload_length.0,
            "file_bytes": shard.file_length.0,
        })).collect::<Vec<_>>(),
        "memory_estimate": plan.memory_estimate,
    })
}
