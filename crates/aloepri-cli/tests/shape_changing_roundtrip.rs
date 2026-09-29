//! Real shape-changing pipeline: source -> Compiler -> padding -> safetensors
//! -> CLI verify. Every assertion is derived from bytes actually written.

mod support;

use aloepri_artifact::{HfArtifact, read_safetensors_header};
use aloepri_core::{
    ModelArtifact, TransformRequest,
    plan::{MethodContract, TransformConfig},
    types::{ByteLength, ByteOffset, TensorName},
};
use serde_json::Value;
use std::{fs, path::Path, process::Command};
use support::shape_change::{PadColumnsExecutor, shape_compiler};
use tempfile::tempdir;

const CONFIG: &[u8] =
    br#"{"model_type":"expand-test","hidden_size":2,"vocab_size":4,"expansion_size":1}"#;

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

fn config() -> TransformConfig {
    TransformConfig {
        method: MethodContract::expand_test(),
        memory_limit: ByteLength(64 * 1024),
        max_shard_size: ByteLength(48),
        ..TransformConfig::default()
    }
}

fn read_tensor(root: &Path, name: &str, length: usize) -> Vec<u8> {
    let artifact = HfArtifact::open(root).unwrap();
    let name = TensorName::try_from(name).unwrap();
    let mut reader = artifact.tensor_reader(&name).unwrap();
    let mut bytes = vec![0_u8; length];
    reader.read_bytes(ByteOffset(0), &mut bytes).unwrap();
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn padding_transform_changes_shape_and_byte_length_and_verifies() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    write_model(&source, CONFIG, &tensors());

    let compiler = shape_compiler(PadColumnsExecutor::default());
    let plan = compiler.plan(&source, &config()).unwrap();

    // The output is planned from output descriptors: the two 2-D tensors grow
    // from 24 to 48 and from 16 to 32 bytes, so the 48-byte shard threshold
    // now splits them, while the source would have fitted in one shard.
    assert_eq!(plan.version, 3);
    assert_eq!(plan.source_inventory.len(), plan.output_inventory.len());
    assert_eq!(plan.output_layout.shards.len(), 2);
    assert!(plan.output_layout.index.is_some());

    let request = TransformRequest {
        input: source.clone(),
        output: output.clone(),
        config: config(),
        resume: false,
    };
    let report = compiler.transform(&request).unwrap();
    assert_eq!(report.tensor_count, 3);
    assert_eq!(report.source_tensor_count, 3);
    assert_eq!(report.payload_bytes, 48 + 32 + 8);
    assert_eq!(report.source_payload_bytes, 24 + 16 + 8);
    assert_ne!(report.payload_bytes, report.source_payload_bytes);
    assert_eq!(report.completed_operations, 3);
    assert!(!report.verification.standard_hf_checkpoint.unwrap());
    assert!(report.verification.runtime_required);

    // AC2: the physical header agrees with the output descriptors.
    for planned in &plan.output_layout.tensors {
        let shard = &plan.output_layout.shards[planned.shard as usize];
        let header = read_safetensors_header(&output.join(&shard.filename)).unwrap();
        let entry = header
            .tensors
            .iter()
            .find(|tensor| tensor.name == planned.name)
            .unwrap();
        assert_eq!(entry.shape, planned.shape);
        assert_eq!(entry.dtype, planned.dtype);
        assert_eq!(entry.byte_length, planned.byte_length);
        assert_eq!(entry.relative_offset, planned.offset.0);
        assert_eq!(planned.byte_length.0, entry.byte_length.0);
    }

    // AC1: the padded rows are the source rows followed by zeros.
    let embedding = read_tensor(&output, "model.embed_tokens.weight", 48);
    assert_eq!(
        embedding,
        f32_bytes(&[1., 2., 0., 0., 3., 4., 0., 0., 5., 6., 0., 0.])
    );
    let renamed = read_tensor(&output, "renamed.target.weight", 32);
    assert_eq!(renamed, f32_bytes(&[7., 8., 0., 0., 9., 10., 0., 0.]));
    assert!(!output.join("model.safetensors").exists());
    let norm = read_tensor(&output, "model.norm.weight", 8);
    assert_eq!(norm, f32_bytes(&[11., 12.]));

    // AC5: the real CLI verifies the diagnostic artifact structurally.
    let verify = Command::new(env!("CARGO_BIN_EXE_aloepri"))
        .args(["verify", output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let outcome: Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(outcome["structure_valid"], true);
    assert_eq!(outcome["manifest_verified"], true);
    assert_eq!(outcome["artifact_valid"], true);
    assert_eq!(outcome["plan_verified"], true);
    // AC6: a non-standard artifact is not reported as vanilla-HF loadable.
    assert_eq!(outcome["standard_hf_checkpoint"], false);
    assert_eq!(outcome["runtime_required"], true);
    assert_eq!(outcome["semantic_verification"], "not_run");
    assert_eq!(outcome["runtime_contract"]["id"], "expand-test-runtime");
}

#[test]
fn pad_columns_streams_rows_larger_than_the_working_buffer() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    // A single row of 4000 bytes (1000 F32 columns) pads to 4008 bytes. With a
    // tiny metadata-and-buffer budget neither the row nor the tensor can be
    // held whole, so the transform only succeeds if both the reader and the
    // sink stream across row and padding boundaries.
    let wide_config =
        br#"{"model_type":"expand-test","hidden_size":1000,"vocab_size":2,"expansion_size":1}"#;
    let values: Vec<f32> = (0..1000).map(|value| value as f32).collect();
    let fixtures: Vec<Fixture> = vec![
        ("model.embed_tokens.weight", vec![1, 1000], values.clone()),
        ("model.norm.weight", vec![2], vec![11., 12.]),
    ];
    write_model(&source, wide_config, &fixtures);

    let budget = TransformConfig {
        method: MethodContract::expand_test(),
        // metadata = 2 inventories * 2 tensors * 256 = 1024; the working buffer
        // is the remaining 1792 bytes, far below a single 4000-byte row.
        memory_limit: ByteLength(2816),
        max_shard_size: ByteLength(4 * 1024 * 1024),
        ..TransformConfig::default()
    };
    let compiler = shape_compiler(PadColumnsExecutor::default());
    let plan = compiler.plan(&source, &budget).unwrap();
    assert!(
        plan.memory_estimate.peak_bytes.0 < 4000,
        "the working set must stay below a single row"
    );
    let report = compiler
        .transform(&TransformRequest {
            input: source.clone(),
            output: output.clone(),
            config: budget,
            resume: false,
        })
        .unwrap();
    assert_eq!(report.payload_bytes, 4008 + 8);

    let padded = read_tensor(&output, "model.embed_tokens.weight", 4008);
    let mut expected = f32_bytes(&values);
    expected.extend_from_slice(&[0_u8; 8]);
    assert_eq!(padded, expected);
}

#[test]
fn padded_artifact_matches_an_independent_row_oracle() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source");
    let output = directory.path().join("output");
    let fixtures = tensors();
    write_model(&source, CONFIG, &fixtures);

    shape_compiler(PadColumnsExecutor::default())
        .transform(&TransformRequest {
            input: source.clone(),
            output: output.clone(),
            config: config(),
            resume: false,
        })
        .unwrap();

    // Independent oracle: read the source rows directly and compare each
    // output row against `source_row || zeros`. Only the two 2-D tensors are
    // padded; the 1-D norm is copied verbatim.
    let source_artifact = HfArtifact::open(&source).unwrap();
    for (name, _, values) in &fixtures {
        if values.len() == 2 {
            continue;
        }
        let padded_name = name.replace(".source.weight", ".target.weight");
        let rows = values.len() / 2;
        let mut reader = source_artifact
            .tensor_reader(&TensorName::try_from(*name).unwrap())
            .unwrap();
        let mut row = vec![0_u8; 8];
        let mut expected = Vec::new();
        for index in 0..rows {
            reader
                .read_bytes(ByteOffset((index * 8) as u64), &mut row)
                .unwrap();
            expected.extend_from_slice(&row);
            expected.extend_from_slice(&[0_u8; 8]);
        }
        assert_eq!(read_tensor(&output, &padded_name, expected.len()), expected);
    }
}
