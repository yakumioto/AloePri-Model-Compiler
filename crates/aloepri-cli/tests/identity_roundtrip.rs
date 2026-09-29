use serde_json::Value;
use std::{fs, path::Path, process::Command};
use tempfile::tempdir;

#[test]
fn identity_round_trip_is_sharded_and_verifiable() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    fs::create_dir(&source).unwrap();
    fs::write(
        source.join("config.json"),
        br#"{"model_type":"llama","tie_word_embeddings":true}"#,
    )
    .unwrap();
    fs::write(
        source.join("generation_config.json"),
        br#"{"eos_token_id":1}"#,
    )
    .unwrap();
    write_safetensors(
        &source.join("model.safetensors"),
        &[("x", vec![1_u8, 2, 3]), ("y", vec![4_u8, 5])],
    );

    let binary = env!("CARGO_BIN_EXE_aloepri");
    let transform = Command::new(binary)
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "4KiB",
            "--max-shard-size",
            "4B",
        ])
        .output()
        .unwrap();
    assert!(
        transform.status.success(),
        "{}",
        String::from_utf8_lossy(&transform.stderr)
    );
    assert!(output.join("model.safetensors.index.json").is_file());
    assert_eq!(
        fs::read(output.join("config.json")).unwrap(),
        fs::read(source.join("config.json")).unwrap()
    );
    assert_eq!(
        fs::read(output.join("generation_config.json")).unwrap(),
        fs::read(source.join("generation_config.json")).unwrap()
    );

    let verify = Command::new(binary)
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let report: Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(report["manifest_verified"], true);
    assert_eq!(report["tensor_count"], 2);
    assert_eq!(report["payload_bytes"], 5);

    let resumed = Command::new(binary)
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "4KiB",
            "--max-shard-size",
            "4B",
            "--resume",
        ])
        .output()
        .unwrap();
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );

    let second = Command::new(binary)
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
        ])
        .output()
        .unwrap();
    assert!(!second.status.success());
}

#[test]
fn indexed_input_can_be_repacked_to_one_shard() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
    write_safetensors(&source.join("part-a.safetensors"), &[("x", vec![1_u8, 2])]);
    write_safetensors(&source.join("part-b.safetensors"), &[("y", vec![3_u8])]);
    fs::write(
        source.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": 3},
            "weight_map": {"x": "part-a.safetensors", "y": "part-b.safetensors"}
        }))
        .unwrap(),
    )
    .unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "4KiB",
            "--max-shard-size",
            "16B",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.join("model.safetensors").is_file());
    assert!(!output.join("model.safetensors.index.json").exists());
}

#[test]
fn unknown_architecture_is_rejected() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"gpt2"}"#).unwrap();
    write_safetensors(&source.join("model.safetensors"), &[("x", vec![1_u8, 2])]);

    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!output.exists());
}

#[test]
fn model_larger_than_memory_limit_still_succeeds() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
    write_safetensors(
        &source.join("model.safetensors"),
        &[("x", vec![7_u8; 6000]), ("y", vec![9_u8; 6000])],
    );

    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "4KiB",
            "--max-shard-size",
            "4GiB",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let verify = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(report["payload_bytes"], 12000);
    assert_eq!(report["manifest_verified"], true);
}

fn write_safetensors(path: &Path, tensors: &[(&str, Vec<u8>)]) {
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
    let mut file = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header_bytes);
    for (_, bytes) in tensors {
        file.extend_from_slice(bytes);
    }
    fs::write(path, file).unwrap();
}
