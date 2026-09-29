use serde_json::Value;
use std::{fs, path::Path, process::Command};
use tempfile::tempdir;

type Fixture = (&'static str, Vec<u64>, Vec<f32>);

const CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":2,"vocab_size":3,"tie_word_embeddings":true}"#;
const UNTIED_CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":2,"vocab_size":3,"tie_word_embeddings":false}"#;

fn tensors() -> Vec<Fixture> {
    vec![
        (
            "model.embed_tokens.weight",
            vec![3, 2],
            vec![1., 2., 3., 4., 5., 6.],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![7., 8.],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![9., 10., 11., 12.],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![13., 14., 15., 16.],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![17., 18., 19., 20.],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![21., 22.],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![23., 24., 25., 26.],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![27., 28., 29., 30.],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![31., 32., 33., 34.],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![35., 36., 37., 38.],
        ),
        ("model.norm.weight", vec![2], vec![39., 40.]),
    ]
}

#[test]
fn token_transform_changes_only_vocab_rows_and_verifies() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    let secret = directory.path().join("client-secret.json");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), CONFIG).unwrap();
    write_safetensors(&source.join("model.safetensors"), &tensors());

    let transform = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--method",
            "aloepri-token",
            "--secret-output",
            secret.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        transform.status.success(),
        "{}",
        String::from_utf8_lossy(&transform.stderr)
    );
    let secret_json: Value = serde_json::from_slice(&fs::read(&secret).unwrap()).unwrap();
    let permutation: Vec<usize> = secret_json["token_permutation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(permutation.len(), 3);
    assert!(
        permutation
            .iter()
            .enumerate()
            .any(|(index, value)| index != *value)
    );
    assert_eq!(
        secret_json["inverse_token_permutation"]
            .as_array()
            .unwrap()
            .len(),
        3
    );

    let source_embedding = tensor_payload(
        &source.join("model.safetensors"),
        "model.embed_tokens.weight",
    );
    let output_embedding = tensor_payload(
        &output.join("model.safetensors"),
        "model.embed_tokens.weight",
    );
    assert_ne!(source_embedding, output_embedding);
    for (original, &obfuscated) in permutation.iter().enumerate() {
        let source_row = &source_embedding[original * 8..(original + 1) * 8];
        let output_row = &output_embedding[obfuscated * 8..(obfuscated + 1) * 8];
        assert_eq!(source_row, output_row);
    }
    assert!(!output.join("lm_head.weight").exists());

    let manifest: Value =
        serde_json::from_slice(&fs::read(output.join("aloepri.json")).unwrap()).unwrap();
    assert_eq!(manifest["artifact_version"], 2);
    assert_eq!(manifest["method"]["id"], "aloepri-token");
    assert_eq!(manifest["secret_id"], secret_json["secret_id"]);
    assert!(manifest.get("token_permutation").is_none());
    assert!(manifest.get("binding_nonce").is_none());

    let verify = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );

    let mut corrupted = secret_json;
    corrupted["token_permutation"][0] =
        Value::from((corrupted["token_permutation"][0].as_u64().unwrap() + 1) % 3);
    fs::write(&secret, serde_json::to_vec(&corrupted).unwrap()).unwrap();
    let resume = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--method",
            "aloepri-token",
            "--secret-output",
            secret.to_str().unwrap(),
            "--resume",
        ])
        .output()
        .unwrap();
    assert!(!resume.status.success());
}

#[test]
fn token_transform_permutates_an_untied_output_projection() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    let secret = directory.path().join("client-secret.json");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), UNTIED_CONFIG).unwrap();
    let mut fixtures = tensors();
    fixtures.push((
        "lm_head.weight",
        vec![3, 2],
        vec![41., 42., 43., 44., 45., 46.],
    ));
    write_safetensors(&source.join("model.safetensors"), &fixtures);

    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--method",
            "aloepri-token",
            "--secret-output",
            secret.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let secret_json: Value = serde_json::from_slice(&fs::read(secret).unwrap()).unwrap();
    let permutation: Vec<usize> = secret_json["token_permutation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect();
    let source_head = tensor_payload(&source.join("model.safetensors"), "lm_head.weight");
    let output_head = tensor_payload(&output.join("model.safetensors"), "lm_head.weight");
    for (original, &obfuscated) in permutation.iter().enumerate() {
        assert_eq!(
            &source_head[original * 8..(original + 1) * 8],
            &output_head[obfuscated * 8..(obfuscated + 1) * 8]
        );
    }
}

#[test]
fn token_transform_accepts_bare_dot_and_absolute_secret_paths() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), CONFIG).unwrap();
    write_safetensors(&source.join("model.safetensors"), &tensors());

    let cases = [
        ("client-secret.json", "bare-output"),
        ("./dot-secret.json", "dot-output"),
    ];
    for (secret_argument, output_name) in cases {
        let output = directory.path().join(output_name);
        let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
            .current_dir(directory.path())
            .args([
                "transform",
                source.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--method",
                "aloepri-token",
                "--secret-output",
                secret_argument,
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "secret path {secret_argument}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(output.join("aloepri.json").is_file());
        assert!(
            directory
                .path()
                .join(secret_argument.trim_start_matches("./"))
                .is_file()
        );
    }

    let absolute_secret = directory.path().join("absolute-secret.json");
    let absolute_output = directory.path().join("absolute-output");
    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .current_dir(directory.path())
        .args([
            "transform",
            source.to_str().unwrap(),
            "--output",
            absolute_output.to_str().unwrap(),
            "--method",
            "aloepri-token",
            "--secret-output",
            absolute_secret.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "absolute secret path: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(absolute_output.join("aloepri.json").is_file());
    assert!(absolute_secret.is_file());
}

fn write_safetensors(path: &Path, tensors: &[Fixture]) {
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!({"format":"pt"}));
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = offset;
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        offset += (values.len() * 4) as u64;
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,offset]}),
        );
    }
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    header_bytes.resize(
        header_bytes.len() + ((8 - header_bytes.len() % 8) % 8),
        b' ',
    );
    let mut file = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&payload);
    fs::write(path, file).unwrap();
}

fn tensor_payload(path: &Path, name: &str) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    let offsets = header[name]["data_offsets"].as_array().unwrap();
    let start = offsets[0].as_u64().unwrap() as usize + 8 + header_length;
    let end = offsets[1].as_u64().unwrap() as usize + 8 + header_length;
    bytes[start..end].to_vec()
}
