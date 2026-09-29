use serde_json::Value;
use std::{fs, path::Path, process::Command};
use tempfile::tempdir;

/// A complete, minimal Llama dense schema: hidden=2, heads=1, kv_heads=1,
/// head_dim=2, intermediate=2, vocab=2, one layer, tied embeddings.
const LLAMA_CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":2,"vocab_size":2,"tie_word_embeddings":true}"#;

type Fixture = (&'static str, Vec<u64>, Vec<u8>);

fn llama_tensors() -> Vec<Fixture> {
    vec![
        ("model.embed_tokens.weight", vec![2, 2], vec![1, 2, 3, 4]),
        ("model.layers.0.input_layernorm.weight", vec![2], vec![5, 6]),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![7, 8, 9, 10],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![11, 12, 13, 14],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![15, 16, 17, 18],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![19, 20],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![21, 22, 23, 24],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![25, 26, 27, 28],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![29, 30, 31, 32],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![33, 34, 35, 36],
        ),
        ("model.norm.weight", vec![2], vec![37, 38]),
    ]
}

fn payload_len(tensors: &[Fixture]) -> u64 {
    tensors.iter().map(|(_, _, bytes)| bytes.len() as u64).sum()
}

#[test]
fn identity_round_trip_is_sharded_and_verifiable() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), LLAMA_CONFIG).unwrap();
    fs::write(
        source.join("generation_config.json"),
        br#"{"eos_token_id":1}"#,
    )
    .unwrap();
    let tensors = llama_tensors();
    write_safetensors(&source.join("model.safetensors"), &tensors);

    let binary = env!("CARGO_BIN_EXE_aloepri");
    let transform = Command::new(binary)
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "8KiB",
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
    assert_eq!(report["tensor_count"], tensors.len());
    assert_eq!(report["payload_bytes"], payload_len(&tensors));

    let resumed = Command::new(binary)
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "8KiB",
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
    fs::write(source.join("config.json"), LLAMA_CONFIG).unwrap();

    let tensors = llama_tensors();
    let (first, rest) = tensors.split_at(5);
    write_safetensors(&source.join("part-a.safetensors"), first);
    write_safetensors(&source.join("part-b.safetensors"), rest);
    let weight_map: serde_json::Map<String, Value> = tensors
        .iter()
        .enumerate()
        .map(|(index, (name, _, _))| {
            let shard = if index < 5 {
                "./part-a.safetensors"
            } else {
                "part-b.safetensors"
            };
            ((*name).to_owned(), Value::String(shard.to_owned()))
        })
        .collect();
    fs::write(
        source.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": payload_len(&tensors)},
            "weight_map": weight_map
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
            "8KiB",
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
    write_safetensors(&source.join("model.safetensors"), &llama_tensors());

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
fn llama_config_without_dimensions_is_rejected() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
    write_safetensors(&source.join("model.safetensors"), &llama_tensors());

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
    fs::write(source.join("config.json"), LLAMA_CONFIG).unwrap();
    let mut tensors = llama_tensors();
    tensors.push(("passthrough.0", vec![6000], vec![7_u8; 6000]));
    tensors.push(("passthrough.1", vec![6000], vec![9_u8; 6000]));
    write_safetensors(&source.join("model.safetensors"), &tensors);

    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--identity",
            "--memory-limit",
            "8KiB",
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
    assert_eq!(report["payload_bytes"], payload_len(&tensors));
    assert_eq!(report["manifest_verified"], true);
}

fn write_safetensors(path: &Path, tensors: &[Fixture]) {
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!({"format":"pt"}));
    for (name, shape, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype":"U8","shape":shape,"data_offsets":[offset,end]}),
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
    for (_, _, bytes) in tensors {
        file.extend_from_slice(bytes);
    }
    fs::write(path, file).unwrap();
}
