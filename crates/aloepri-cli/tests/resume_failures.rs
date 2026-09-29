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

fn prepare(source: &Path) {
    fs::create_dir(source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
    write_safetensors(source.join("model.safetensors"));
}

fn write_safetensors(path: std::path::PathBuf) {
    let bytes = vec![1_u8, 2, 3];
    let mut header = serde_json::Map::new();
    header.insert(
        "x".into(),
        serde_json::json!({"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}),
    );
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
    file.write_all(&bytes).unwrap();
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
