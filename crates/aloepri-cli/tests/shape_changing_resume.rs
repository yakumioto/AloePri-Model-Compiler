//! Resume semantics for the shape-changing pipeline: an interrupted staging is
//! resumed only by a byte-identical contract, and every incompatible variant
//! fails closed before the staging directory is touched.

mod support;

use aloepri_core::{
    TransformRequest,
    plan::{MethodContract, TransformConfig},
    types::ByteLength,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use support::shape_change::{
    FaultInjectingExecutor, PadColumnsExecutor, Tamper, shape_compiler, shape_compiler_with,
};
use tempfile::tempdir;

const CONFIG: &[u8] =
    br#"{"model_type":"expand-test","hidden_size":2,"vocab_size":4,"expansion_size":1}"#;
const WIDER_CONFIG: &[u8] =
    br#"{"model_type":"expand-test","hidden_size":2,"vocab_size":4,"expansion_size":2}"#;

type Fixture = (&'static str, Vec<u64>, Vec<f32>);

fn tensors() -> Vec<Fixture> {
    vec![
        (
            "model.embed_tokens.weight",
            vec![3, 2],
            vec![1., 2., 3., 4., 5., 6.],
        ),
        ("model.norm.weight", vec![2], vec![11., 12.]),
        ("renamed.source.weight", vec![2, 2], vec![7., 8., 9., 10.]),
    ]
}

fn write_model(root: &Path, config: &[u8], fixtures: &[Fixture]) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("config.json"), config).unwrap();
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let mut payload = Vec::new();
    for (name, shape, values) in fixtures {
        let start = offset;
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        offset += (values.len() * 4) as u64;
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, offset]}),
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
    fs::write(root.join("model.safetensors"), file).unwrap();
}

fn base_config() -> TransformConfig {
    TransformConfig {
        method: MethodContract::expand_test(),
        memory_limit: ByteLength(64 * 1024),
        max_shard_size: ByteLength(48),
        ..TransformConfig::default()
    }
}

/// A recursive snapshot: relative path -> file bytes (or `None` for a
/// directory). Comparing it proves nothing under the staging directory was
/// created, resized or edited.
fn snapshot(root: &Path) -> BTreeMap<String, Option<Vec<u8>>> {
    let mut entries = BTreeMap::new();
    if !root.exists() {
        return entries;
    }
    let mut stack = vec![root.to_owned()];
    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(&current).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if path.is_dir() {
                entries.insert(format!("{relative}/"), None);
                stack.push(path);
            } else {
                entries.insert(relative, Some(fs::read(&path).unwrap()));
            }
        }
    }
    entries
}

fn locks(directory: &Path) -> Vec<String> {
    fs::read_dir(directory)
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            name.ends_with(".aloepri.lock").then_some(name)
        })
        .collect()
}

fn manifest_tensors(root: &Path) -> Vec<Value> {
    let manifest = fs::read(root.join("aloepri.json")).unwrap();
    let parsed: Value = serde_json::from_slice(&manifest).unwrap();
    parsed["tensors"].as_array().unwrap().clone()
}

fn completed_count(work: &Path) -> usize {
    let bytes = fs::read(work.join("checkpoint.json")).unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    parsed["completed"].as_object().unwrap().len()
}

fn run(
    tamper: Tamper,
    input: &Path,
    output: &Path,
    config: TransformConfig,
    resume: bool,
) -> aloepri_core::Result<aloepri_core::TransformReport> {
    shape_compiler_with(tamper, PadColumnsExecutor::default()).transform(&TransformRequest {
        input: input.to_owned(),
        output: output.to_owned(),
        config,
        resume,
    })
}

#[test]
fn interrupted_padding_staging_resumes_to_the_clean_result() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let clean = directory.path().join("clean");
    let resumed = directory.path().join("resumed");
    write_model(&source, CONFIG, &tensors());

    shape_compiler(PadColumnsExecutor::default())
        .transform(&TransformRequest {
            input: source.clone(),
            output: clean.clone(),
            config: base_config(),
            resume: false,
        })
        .unwrap();
    let clean_manifest = manifest_tensors(&clean);

    // Fail while writing the second operation: the first is recorded complete
    // and the shards are allocated, but nothing is published.
    let failed = shape_compiler(FaultInjectingExecutor::new(1)).transform(&TransformRequest {
        input: source.clone(),
        output: resumed.clone(),
        config: base_config(),
        resume: false,
    });
    assert!(failed.is_err());
    assert!(!resumed.exists(), "a failed run must not publish");
    let work = directory.path().join(".resumed.aloepri-work");
    assert_eq!(completed_count(&work), 1);

    let resumed_report = run(Tamper::None, &source, &resumed, base_config(), true).unwrap();
    assert_eq!(resumed_report.completed_operations, 3);
    assert_eq!(manifest_tensors(&resumed), clean_manifest);
    assert!(!work.exists());
}

#[test]
fn incompatible_resume_contracts_fail_before_touching_staging() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let clean = directory.path().join("clean");
    let resumed = directory.path().join("resumed");
    write_model(&source, CONFIG, &tensors());
    shape_compiler(PadColumnsExecutor::default())
        .transform(&TransformRequest {
            input: source.clone(),
            output: clean.clone(),
            config: base_config(),
            resume: false,
        })
        .unwrap();
    let clean_manifest = manifest_tensors(&clean);

    let failed = shape_compiler(FaultInjectingExecutor::new(1)).transform(&TransformRequest {
        input: source.clone(),
        output: resumed.clone(),
        config: base_config(),
        resume: false,
    });
    assert!(failed.is_err());
    let work = directory.path().join(".resumed.aloepri-work");
    let before = snapshot(&work);
    let locks_before = locks(directory.path());

    // A wider source: a different expansion/physical size and fingerprint.
    let wider = directory.path().join("wider");
    write_model(&wider, WIDER_CONFIG, &tensors());

    // A source whose tensor payload changed but whose config did not.
    let edited = directory.path().join("edited");
    write_model(&edited, CONFIG, &tensors());
    let mut bytes = fs::read(edited.join("model.safetensors")).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(edited.join("model.safetensors"), &bytes).unwrap();

    let mut shorter_shards = base_config();
    shorter_shards.max_shard_size = ByteLength(24);

    let mut other_method = base_config();
    other_method.method = MethodContract::identity();

    let variants: Vec<(&str, Tamper, PathBuf, TransformConfig)> = vec![
        (
            "expansion and physical size",
            Tamper::None,
            wider.clone(),
            base_config(),
        ),
        (
            "source fingerprint",
            Tamper::None,
            edited.clone(),
            base_config(),
        ),
        ("method", Tamper::None, source.clone(), other_method),
        ("shard layout", Tamper::None, source.clone(), shorter_shards),
        (
            "runtime version",
            Tamper::RuntimeVersion,
            source.clone(),
            base_config(),
        ),
        (
            "physical dimensions",
            Tamper::ExtraPhysicalDimension,
            source.clone(),
            base_config(),
        ),
        (
            "standard flag",
            Tamper::StandardFlag,
            source.clone(),
            base_config(),
        ),
    ];

    for (label, tamper, input, config) in variants {
        // Every variant resumes the same interrupted staging, so a mismatch
        // must be detected rather than silently starting a fresh compile.
        let result = run(tamper, &input, &resumed, config, true);
        assert!(result.is_err(), "{label} resume must be rejected");
        assert!(
            !resumed.exists(),
            "{label} rejection must not publish an output"
        );
        assert_eq!(
            snapshot(&work),
            before,
            "{label} rejection must leave staging byte-identical"
        );
        assert_eq!(
            locks(directory.path()),
            locks_before,
            "{label} rejection must not leave a new lock"
        );
    }

    // The original contract still finishes the interrupted run.
    let report = run(Tamper::None, &source, &resumed, base_config(), true).unwrap();
    assert_eq!(report.completed_operations, 3);
    assert_eq!(manifest_tensors(&resumed), clean_manifest);
}

#[test]
fn incomplete_or_corrupt_completion_records_are_rejected() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let resumed = directory.path().join("resumed");
    write_model(&source, CONFIG, &tensors());

    let failed = shape_compiler(FaultInjectingExecutor::new(1)).transform(&TransformRequest {
        input: source.clone(),
        output: resumed.clone(),
        config: base_config(),
        resume: false,
    });
    assert!(failed.is_err());
    let work = directory.path().join(".resumed.aloepri-work");
    let checkpoint = work.join("checkpoint.json");

    // A completed map with a gap is not a contiguous prefix.
    let original: Value = serde_json::from_slice(&fs::read(&checkpoint).unwrap()).unwrap();
    let mut gapped = original.clone();
    let completed = gapped["completed"].as_object_mut().unwrap();
    let first = completed.keys().next().unwrap().clone();
    let digest = completed[&first].clone();
    completed.clear();
    completed.insert("0".into(), digest.clone());
    completed.insert("2".into(), digest);
    fs::write(&checkpoint, serde_json::to_vec(&gapped).unwrap()).unwrap();
    assert!(
        run(Tamper::None, &source, &resumed, base_config(), true).is_err(),
        "a non-prefix completed map must be rejected"
    );
    assert!(!resumed.exists());

    // A completed entry whose recorded digest no longer matches the tensor.
    let mut corrupted = original.clone();
    corrupted["completed"][&first] = Value::from("00".repeat(32));
    fs::write(&checkpoint, serde_json::to_vec(&corrupted).unwrap()).unwrap();
    assert!(
        run(Tamper::None, &source, &resumed, base_config(), true).is_err(),
        "a mismatched completed digest must be rejected"
    );
    assert!(!resumed.exists());
}
