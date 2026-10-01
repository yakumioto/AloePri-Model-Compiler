use anyhow::{Result, anyhow};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "aloepri",
    version,
    about = "Identity-preserving Hugging Face safetensors compiler"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Inspect { model: PathBuf },
    Plan(PlanArgs),
    Transform(TransformArgs),
    Verify { model: PathBuf },
}

#[derive(Debug, Args)]
pub struct PlanArgs {
    pub model: PathBuf,
    #[arg(long, conflicts_with = "method")]
    pub identity: bool,
    #[arg(long)]
    pub method: Option<String>,
    #[arg(long, default_value = "256MiB", value_parser = parse_size)]
    pub memory_limit: u64,
    #[arg(long, default_value = "4GiB", value_parser = parse_size)]
    pub max_shard_size: u64,
}

#[derive(Debug, Args)]
pub struct TransformArgs {
    pub model: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, conflicts_with = "method")]
    pub identity: bool,
    #[arg(long)]
    pub method: Option<String>,
    #[arg(long)]
    pub secret_output: Option<PathBuf>,
    #[arg(long)]
    pub expansion_size: Option<u64>,
    #[arg(long, allow_hyphen_values = true)]
    pub keymat_lambda: Option<f64>,
    #[arg(
        long,
        help = "Deterministic fixture seed; never use for private production keys"
    )]
    pub keymat_fixture_seed: Option<u64>,
    #[arg(long, value_parser = ["algorithm1-v1", "algorithm1-balanced-null-v2", "algorithm1-signed-null-v3"])]
    pub keymat_algorithm: Option<String>,
    #[arg(long, default_value = "256MiB", value_parser = parse_size)]
    pub memory_limit: u64,
    #[arg(long, default_value = "4GiB", value_parser = parse_size)]
    pub max_shard_size: u64,
    #[arg(long)]
    pub resume: bool,
}

pub fn parse_size(value: &str) -> Result<u64, String> {
    let value = value.trim();
    let (number, multiplier) = [
        ("GiB", 1024_u64 * 1024 * 1024),
        ("MiB", 1024_u64 * 1024),
        ("KiB", 1024_u64),
        ("GB", 1000_u64 * 1000 * 1000),
        ("MB", 1000_u64 * 1000),
        ("KB", 1000_u64),
        ("B", 1_u64),
    ]
    .into_iter()
    .find_map(|(suffix, multiplier)| {
        value
            .strip_suffix(suffix)
            .map(|number| (number, multiplier))
    })
    .unwrap_or((value, 1));
    let number = number
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("invalid size {value}"))?;
    number
        .checked_mul(multiplier)
        .ok_or_else(|| format!("size {value} overflows"))
}

pub fn resolve_method(
    identity: bool,
    method: Option<&str>,
) -> Result<aloepri_core::MethodContract> {
    if identity && method.is_some() {
        return Err(anyhow!("--identity and --method cannot be used together"));
    }
    match (identity, method) {
        (true, None) | (false, Some("identity")) => Ok(aloepri_core::MethodContract::identity()),
        (false, Some("aloepri-token")) => Ok(aloepri_core::MethodContract::aloepri_token()),
        (false, Some("aloepri-keymat")) => Ok(aloepri_core::MethodContract::aloepri_keymat()),
        (false, Some(other)) => Err(anyhow!("unsupported transform method {other}")),
        (false, None) => Err(anyhow!("an explicit --identity or --method is required")),
        (true, Some(_)) => unreachable!("method conflict is checked above"),
    }
}
