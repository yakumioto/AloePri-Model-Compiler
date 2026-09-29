use aloepri_architecture::Registry;
use aloepri_artifact::{HfBackend, StreamingWriter};
use aloepri_core::{Compiler, TransformConfig, plan::TransformPlan, types::ByteLength};
use aloepri_transform::IdentityExecutor;
use serde_json::Value;
use std::{fs, io::Seek, path::Path, process::Command};
use tempfile::tempdir;

#[test]
fn tampered_output_is_rejected() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    prepare(&source);

    run(&[
        "transform",
        path(&source),
        "--output",
        path(&output),
        "--identity",
    ]);
    assert!(run(&["verify", path(&output)]).status.success());

    let shard = output.join("model.safetensors");
    let mut bytes = fs::read(&shard).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&shard, &bytes).unwrap();

    let verify = run(&["verify", path(&output)]);
    assert!(!verify.status.success());

    let resume = run(&[
        "transform",
        path(&source),
        "--output",
        path(&output),
        "--identity",
        "--resume",
    ]);
    assert!(!resume.status.success());
}

#[test]
fn checkpoint_contract_mismatch_blocks_resume() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    prepare(&source);

    let work = directory.path().join(".output.aloepri-work");
    fs::create_dir_all(work.join("artifact")).unwrap();
    fs::write(
        work.join("checkpoint.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "source_fingerprint": "00".repeat(32),
            "plan_hash": "11".repeat(32),
            "output_layout_hash": "22".repeat(32),
            "method": {"id": "identity", "version": "0.1"},
            "secret_key_id": null,
            "completed": {}
        }))
        .unwrap(),
    )
    .unwrap();

    let resume = run(&[
        "transform",
        path(&source),
        "--output",
        path(&output),
        "--identity",
        "--resume",
    ]);
    assert!(!resume.status.success());
    assert!(!output.exists());
}

#[test]
fn unsupported_checkpoint_schema_is_rejected() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    prepare(&source);

    let work = directory.path().join(".output.aloepri-work");
    fs::create_dir_all(work.join("artifact")).unwrap();
    fs::write(
        work.join("checkpoint.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 99,
            "source_fingerprint": "00".repeat(32),
            "plan_hash": "11".repeat(32),
            "output_layout_hash": "22".repeat(32),
            "method": {"id": "identity", "version": "0.1"},
            "secret_key_id": null,
            "completed": {}
        }))
        .unwrap(),
    )
    .unwrap();

    assert!(
        !run(&[
            "transform",
            path(&source),
            "--output",
            path(&output),
            "--identity",
            "--resume",
        ])
        .status
        .success()
    );
}

#[test]
fn interrupted_sharded_staging_resumes_to_the_clean_result() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let clean = directory.path().join("clean");
    let resumed = directory.path().join("resumed");
    prepare_sharded_input(&source);

    let clean_run = run(&[
        "transform",
        path(&source),
        "--output",
        path(&clean),
        "--identity",
        "--memory-limit",
        "4KiB",
        "--max-shard-size",
        "4B",
    ]);
    assert!(
        clean_run.status.success(),
        "{}",
        String::from_utf8_lossy(&clean_run.stderr)
    );
    let clean_manifest = manifest_tensors(&clean);
    assert!(
        clean_manifest.len() >= 2,
        "fixture must produce several tensors"
    );

    // Reproduce the crash state: the writer created and sized every planned
    // shard, then the process died before the index and manifest were written.
    let work = directory.path().join(".resumed.aloepri-work");
    let candidate = work.join("artifact");
    fs::create_dir_all(&candidate).unwrap();
    let plan = build_plan(&source, 4 * 1024, 4);
    let writer = StreamingWriter::create(&candidate, plan.output_layout.clone()).unwrap();
    drop(writer);
    assert!(candidate.join("model-00001-of-00002.safetensors").is_file());
    assert!(!candidate.join("model.safetensors").exists());
    assert!(!candidate.join("model.safetensors.index.json").exists());
    aloepri_core::backend::ArtifactBackend::store_checkpoint(
        &HfBackend,
        &work.join("checkpoint.json"),
        &plan,
        &std::collections::BTreeMap::new(),
    )
    .unwrap();

    let resume_run = run(&[
        "transform",
        path(&source),
        "--output",
        path(&resumed),
        "--identity",
        "--memory-limit",
        "4KiB",
        "--max-shard-size",
        "4B",
        "--resume",
    ]);
    assert!(
        resume_run.status.success(),
        "resume failed: {}",
        String::from_utf8_lossy(&resume_run.stderr)
    );
    assert!(resumed.join("model.safetensors.index.json").is_file());
    assert_eq!(manifest_tensors(&resumed), clean_manifest);

    let report: Value = serde_json::from_slice(&resume_run.stdout).unwrap();
    assert_eq!(report["verification"]["manifest_verified"], true);
}

fn build_plan(source: &Path, memory_limit: u64, max_shard_size: u64) -> TransformPlan {
    let compiler: Compiler<HfBackend, Registry, IdentityExecutor> =
        Compiler::new(HfBackend, Registry::new(), IdentityExecutor::default());
    let config = TransformConfig {
        memory_limit: ByteLength(memory_limit),
        max_shard_size: ByteLength(max_shard_size),
        ..TransformConfig::default()
    };
    compiler.plan(source, &config).unwrap()
}

fn manifest_tensors(root: &Path) -> Vec<Value> {
    let manifest = fs::read(root.join("aloepri.json")).unwrap();
    let parsed: Value = serde_json::from_slice(&manifest).unwrap();
    parsed["tensors"].as_array().unwrap().clone()
}

fn prepare(source: &Path) {
    fs::create_dir(source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
    write_safetensors(source.join("model.safetensors"), &[("x", vec![1_u8, 2, 3])]);
}

fn prepare_sharded_input(source: &Path) {
    fs::create_dir(source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
    write_safetensors(
        source.join("model.safetensors"),
        &[("x", vec![1_u8, 2, 3]), ("y", vec![4_u8, 5])],
    );
}

fn write_safetensors(path: std::path::PathBuf, tensors: &[(&str, Vec<u8>)]) {
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!({"format":"pt"}));
    for (name, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype":"U8","shape":[bytes.len()],"data_offsets":[offset,end]}),
        );
        offset = end;
    }
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    header_bytes.resize(
        header_bytes.len() + ((8 - header_bytes.len() % 8) % 8),
        b' ',
    );
    let mut file = fs::File::create(path).unwrap();
    use std::io::Write;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    for (_, bytes) in tensors {
        file.write_all(bytes).unwrap();
    }
    file.rewind().unwrap();
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args(args)
        .output()
        .unwrap()
}

fn path(value: &Path) -> &str {
    value.to_str().unwrap()
}
