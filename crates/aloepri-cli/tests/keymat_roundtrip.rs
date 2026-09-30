use aloepri_architecture::Registry;
use aloepri_artifact::{HfArtifact, HfBackend};
use aloepri_core::{
    ByteLength, Compiler, CompilerError, MemoryBudget, MethodContract, ModelArtifact, Operation,
    Result, TensorSink, TransformConfig, TransformExecutor, TransformRequest,
};
use aloepri_secret::keymat::KeyMatSecretV1;
use aloepri_transform::KeyMatExecutor;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, process::Command, sync::Arc};
use tempfile::tempdir;

fn source(root: &Path, tied: bool, head: bool, dtype: &str) {
    fs::create_dir(root).unwrap();
    fs::write(root.join("config.json"),serde_json::to_vec(&json!({"model_type":"llama","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":7,"vocab_size":11,"tie_word_embeddings":tied})).unwrap()).unwrap();
    let dimensions=serde_json::from_value(json!({"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":7,"vocab_size":11})).unwrap();
    let specs = aloepri_core::keymat::tensor_specs(&dimensions, 8).unwrap();
    let mut header = serde_json::Map::new();
    let mut payload = vec![];
    for (name, (shape, _, _)) in specs {
        if name == "lm_head.weight" && !head {
            continue;
        }
        let start = payload.len();
        for i in 0..shape.iter().product::<u64>() {
            let value = if name.contains("norm") {
                1.0
            } else {
                (i as f32 - 7.0) / 31.0
            };
            match dtype {
                "F32" => payload.extend_from_slice(&value.to_le_bytes()),
                "F16" | "BF16" => payload.extend_from_slice(&0u16.to_le_bytes()),
                _ => payload.push(0),
            }
        }
        header.insert(
            name,
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let mut bytes = serde_json::to_vec(&header).unwrap();
    bytes.resize(bytes.len().div_ceil(8) * 8, b' ');
    let mut all = (bytes.len() as u64).to_le_bytes().to_vec();
    all.extend_from_slice(&bytes);
    all.extend_from_slice(&payload);
    fs::write(root.join("model.safetensors"), all).unwrap();
}

fn cli(input: &Path, output: &Path, secret: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args([
            "transform",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--method",
            "aloepri-keymat",
            "--expansion-size",
            "2",
            "--keymat-lambda",
            "0.3",
            "--keymat-fixture-seed",
            "42",
            "--secret-output",
            secret.to_str().unwrap(),
            "--max-shard-size",
            "256B",
        ])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn cli_roundtrip_materializes_head_and_preserves_logical_config() {
    for (tied, head) in [(true, false), (true, true), (false, true)] {
        let dir = tempdir().unwrap();
        let input = dir.path().join("source");
        let output = dir.path().join("output");
        let secret = dir.path().join("secret.json");
        source(&input, tied, head, "F32");
        let result = cli(&input, &output, &secret, &[]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            fs::read(input.join("config.json")).unwrap(),
            fs::read(output.join("config.json")).unwrap()
        );
        let artifact = HfArtifact::open(&output).unwrap();
        assert_eq!(artifact.tensors().len(), 12);
        let manifest: Value =
            serde_json::from_slice(&fs::read(output.join("aloepri.json")).unwrap()).unwrap();
        assert_eq!(manifest["artifact_version"], 4);
        let text = manifest.to_string();
        for private in ["master_seed", "p_digest", "q_digest", "key-material.bin"] {
            assert!(!text.contains(private));
        }
        let verify = Command::new(env!("CARGO_BIN_EXE_aloepri"))
            .args(["verify", output.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(verify.status.success());
        let flags: Value = serde_json::from_slice(&verify.stdout).unwrap();
        for flag in [
            "structure_valid",
            "manifest_verified",
            "plan_verified",
            "runtime_required",
        ] {
            assert_eq!(flags[flag], true);
        }
        assert_eq!(flags["standard_hf_checkpoint"], false);
        assert_eq!(flags["semantic_verification"], "not_run");
        let (_, keys) = KeyMatSecretV1::read(&secret, u64::MAX).unwrap();
        let mut embedding = artifact
            .tensor_reader(
                &aloepri_core::TensorName::try_from("model.embed_tokens.weight").unwrap(),
            )
            .unwrap();
        let mut bytes = vec![0; 11 * 8 * 4];
        embedding
            .read_bytes(aloepri_core::ByteOffset(0), &mut bytes)
            .unwrap();
        for row in 0..11 {
            for col in 0..8 {
                let expected: f64 = (0..4)
                    .map(|k| ((row * 4 + k) as f32 - 7.0) as f64 / 31.0 * keys.p()[k * 8 + col])
                    .sum();
                let actual = f32::from_le_bytes(
                    bytes[(row * 8 + col) * 4..(row * 8 + col + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                assert!((actual as f64 - expected).abs() < 1e-6);
            }
        }
        let resumed = cli(&input, &output, &secret, &["--resume"]);
        assert!(resumed.status.success());
    }
}

#[test]
fn bare_secret_filename_is_synced_before_publication() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("source");
    let output = dir.path().join("output");
    source(&input, true, false, "F32");
    let result = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .current_dir(dir.path())
        .args([
            "transform",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--method",
            "aloepri-keymat",
            "--expansion-size",
            "2",
            "--keymat-lambda",
            "0.3",
            "--secret-output",
            "secret.json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.join("aloepri.json").is_file());
    KeyMatSecretV1::read(&dir.path().join("secret.json"), u64::MAX).unwrap();
}

#[test]
fn unsupported_dtype_and_low_budget_leave_no_secret_or_staging() {
    for dtype in ["F16", "BF16", "U8", "F32"] {
        let dir = tempdir().unwrap();
        let input = dir.path().join("source");
        let output = dir.path().join("output");
        let secret = dir.path().join("secret.json");
        source(&input, true, false, dtype);
        let result = cli(&input, &output, &secret, &["--memory-limit", "1KiB"]);
        assert!(!result.status.success());
        assert!(!output.exists());
        assert!(!secret.exists());
        assert!(!dir.path().join("key-material.bin").exists());
        assert!(!dir.path().join(".output.aloepri-work").exists());
        if dtype != "F32" {
            assert!(String::from_utf8_lossy(&result.stderr).contains("unsupported"));
        }
    }
}

struct Fault(KeyMatExecutor);
impl TransformExecutor for Fault {
    fn requirements(&self, c: &TransformConfig) -> Result<()> {
        self.0.requirements(c)
    }
    fn execute_operation(
        &self,
        a: &dyn ModelArtifact,
        o: &Operation,
        s: &mut dyn TensorSink,
        b: &MemoryBudget,
    ) -> Result<()> {
        if o.id.0 == 2 {
            return Err(CompilerError::Invariant("injected interruption".into()));
        }
        self.0.execute_operation(a, o, s, b)
    }
}
fn snapshot(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    let mut stack = vec![root.to_owned()];
    while let Some(path) = stack.pop() {
        for e in fs::read_dir(path).unwrap() {
            let p = e.unwrap().path();
            let name = p.strip_prefix(root).unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                map.insert(format!("{name}/"), vec![]);
                stack.push(p);
            } else {
                map.insert(name, fs::read(p).unwrap());
            }
        }
    }
    map
}
#[test]
fn resume_mismatches_leave_staging_and_locks_byte_identical() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("source");
    let output = dir.path().join("output");
    source(&input, true, false, "F32");
    let artifact = HfArtifact::open(&input).unwrap();
    let fingerprint = artifact.fingerprint().unwrap();
    let (secret, keys, _) =
        KeyMatSecretV1::generate(fingerprint, 4, 2, 0.3, Some([42; 32])).unwrap();
    let keys = Arc::new(keys);
    let config = TransformConfig {
        method: MethodContract::aloepri_keymat(),
        keymat_binding: Some(secret.binding().unwrap()),
        memory_limit: ByteLength(64 * 1024),
        max_shard_size: ByteLength(128),
        ..TransformConfig::default()
    };
    let mut request = TransformRequest {
        input: input.clone(),
        output: output.clone(),
        config: config.clone(),
        resume: false,
    };
    assert!(
        Compiler::new(
            HfBackend,
            Registry::new(),
            Fault(KeyMatExecutor::new(&secret, keys.clone()).unwrap())
        )
        .transform(&request)
        .is_err()
    );
    let before = snapshot(dir.path());
    for (name, bytes) in &before {
        if name.starts_with(".output.aloepri-work/") {
            assert!(!name.ends_with("key-material.bin"));
            if name.ends_with(".json") {
                let text = String::from_utf8_lossy(bytes);
                for private in ["master_seed", "p_digest", "q_digest", "key-material.bin"] {
                    assert!(!text.contains(private));
                }
            }
        }
    }
    request.resume = true;
    let compiler = Compiler::new(
        HfBackend,
        Registry::new(),
        KeyMatExecutor::new(&secret, keys.clone()).unwrap(),
    );
    for (h, lambda, seed) in [(4, 0.3, [42; 32]), (2, 0.4, [42; 32]), (2, 0.3, [41; 32])] {
        let (other, material, _) =
            KeyMatSecretV1::generate(fingerprint, 4, h, lambda, Some(seed)).unwrap();
        let mut mismatch = request.clone();
        mismatch.config.keymat_binding = Some(other.binding().unwrap());
        let other_compiler = Compiler::new(
            HfBackend,
            Registry::new(),
            KeyMatExecutor::new(&other, Arc::new(material)).unwrap(),
        );
        assert!(other_compiler.transform(&mismatch).is_err());
        assert_eq!(before, snapshot(dir.path()));
    }
    let mut mismatch = request.clone();
    mismatch.config.max_shard_size = ByteLength(256);
    assert!(compiler.transform(&mismatch).is_err());
    assert_eq!(before, snapshot(dir.path()));
    let checkpoint = dir.path().join(".output.aloepri-work/checkpoint.json");
    let original = fs::read(&checkpoint).unwrap();
    for field in [
        "physical", "mode", "version", "source", "schema", "secret", "plan",
    ] {
        let mut json: Value = serde_json::from_slice(&original).unwrap();
        match field {
            "physical" => json["runtime_contract"]["physical_dimensions"]["hidden_size"] = json!(9),
            "mode" => json["runtime_contract"]["norm_mode"] = json!("paper"),
            "version" => json["runtime_contract"]["version"] = json!("2"),
            "source" => json["source_fingerprint"] = json!("00".repeat(32)),
            "schema" => json["schema_version"] = json!(3),
            "secret" => json["secret_id"] = json!("00".repeat(32)),
            _ => json["plan_hash"] = json!("00".repeat(32)),
        }
        fs::write(&checkpoint, serde_json::to_vec(&json).unwrap()).unwrap();
        let changed = snapshot(dir.path());
        assert!(compiler.transform(&request).is_err());
        assert_eq!(changed, snapshot(dir.path()));
    }
    fs::write(&checkpoint, original).unwrap();
    let source_file = input.join("model.safetensors");
    let original_source = fs::read(&source_file).unwrap();
    let mut changed_source = original_source.clone();
    *changed_source.last_mut().unwrap() ^= 1;
    fs::write(&source_file, changed_source).unwrap();
    let changed = snapshot(dir.path());
    assert!(compiler.transform(&request).is_err());
    assert_eq!(changed, snapshot(dir.path()));
    fs::write(&source_file, original_source).unwrap();
    assert!(compiler.transform(&request).is_ok());
    let (other, material, _) =
        KeyMatSecretV1::generate(fingerprint, 4, 2, 0.3, Some([41; 32])).unwrap();
    request.config.keymat_binding = Some(other.binding().unwrap());
    let other_compiler = Compiler::new(
        HfBackend,
        Registry::new(),
        KeyMatExecutor::new(&other, Arc::new(material)).unwrap(),
    );
    let published = snapshot(dir.path());
    assert!(other_compiler.transform(&request).is_err());
    assert_eq!(published, snapshot(dir.path()));
}
