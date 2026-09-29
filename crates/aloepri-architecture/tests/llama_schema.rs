use aloepri_architecture::{LlamaDenseAdapter, Registry};
use aloepri_core::{
    error::{CompilerError, Result},
    io::TensorReader,
    model::{ArchitectureAdapter, ArchitectureRegistry, ModelArtifact, ModelSpec},
    plan::TransformConfig,
    types::{
        ByteLength, ByteOffset, DType, ModelFingerprint, ShardId, TensorDescriptor, TensorLocation,
        TensorName, TensorShape,
    },
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

struct MockArtifact {
    spec: ModelSpec,
}

impl MockArtifact {
    fn new(config: Value, tensors: Vec<TensorDescriptor>) -> Self {
        let aliases = if config.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true)
            && !tensors
                .iter()
                .any(|tensor| tensor.name.as_str() == "lm_head.weight")
        {
            BTreeMap::from([(
                "lm_head.weight".to_owned(),
                "model.embed_tokens.weight".to_owned(),
            )])
        } else {
            BTreeMap::new()
        };
        let architecture = config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        Self {
            spec: ModelSpec {
                architecture,
                config,
                tensors,
                aliases,
            },
        }
    }
}

impl ModelArtifact for MockArtifact {
    fn root(&self) -> &Path {
        Path::new(".")
    }

    fn config_bytes(&self) -> &[u8] {
        &[]
    }

    fn config(&self) -> &Value {
        &self.spec.config
    }

    fn model_spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn tensor_reader(&self, name: &TensorName) -> Result<Box<dyn TensorReader>> {
        Err(CompilerError::MissingTensor {
            name: name.to_string(),
        })
    }

    fn fingerprint(&self) -> Result<ModelFingerprint> {
        Err(CompilerError::Unsupported("mock artifact".into()))
    }
}

fn tensor(name: &str, shape: &[u64]) -> TensorDescriptor {
    let elements = shape.iter().product::<u64>();
    TensorDescriptor {
        name: TensorName::try_from(name).unwrap(),
        shape: TensorShape::new(shape.to_vec()),
        dtype: DType::U8,
        byte_length: ByteLength(elements),
        location: TensorLocation {
            shard: ShardId(0),
            offset: ByteOffset(0),
            length: ByteLength(elements),
        },
    }
}

/// hidden=4, heads=2, kv_heads=1, head_dim=2, intermediate=8, vocab=4, 1 layer.
fn llama_config() -> Value {
    json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "vocab_size": 4,
        "tie_word_embeddings": true
    })
}

fn llama_tensors() -> Vec<TensorDescriptor> {
    let mut tensors = vec![
        tensor("model.layers.0.self_attn.q_proj.weight", &[4, 4]),
        tensor("model.layers.0.self_attn.k_proj.weight", &[2, 4]),
        tensor("model.layers.0.self_attn.v_proj.weight", &[2, 4]),
        tensor("model.layers.0.self_attn.o_proj.weight", &[4, 4]),
        tensor("model.layers.0.mlp.gate_proj.weight", &[8, 4]),
        tensor("model.layers.0.mlp.up_proj.weight", &[8, 4]),
        tensor("model.layers.0.mlp.down_proj.weight", &[4, 8]),
        tensor("model.layers.0.input_layernorm.weight", &[4]),
        tensor("model.layers.0.post_attention_layernorm.weight", &[4]),
        tensor("model.norm.weight", &[4]),
        tensor("model.embed_tokens.weight", &[4, 4]),
    ];
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    tensors
}

fn build(config: Value, tensors: Vec<TensorDescriptor>) -> Result<()> {
    LlamaDenseAdapter
        .build_plan(
            &MockArtifact::new(config, tensors),
            &TransformConfig::default(),
        )
        .map(|_| ())
}

#[test]
fn accepts_a_complete_llama_schema() {
    build(llama_config(), llama_tensors()).unwrap();
}

#[test]
fn accepts_the_documented_head_fallbacks() {
    // Without `num_key_value_heads` the key/value width follows the head count;
    // without `head_dim` it follows hidden_size / num_attention_heads.
    let mut config = llama_config();
    config
        .as_object_mut()
        .unwrap()
        .remove("num_key_value_heads");
    let mut tensors = llama_tensors();
    for tensor in &mut tensors {
        if tensor.name.as_str().ends_with("k_proj.weight")
            || tensor.name.as_str().ends_with("v_proj.weight")
        {
            tensor.shape = TensorShape::new(vec![4, 4]);
            tensor.byte_length = ByteLength(16);
        }
    }
    build(config, tensors).unwrap();
}

#[test]
fn rejects_a_config_missing_required_dimensions() {
    for key in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
    ] {
        let mut config = llama_config();
        config.as_object_mut().unwrap().remove(key);
        let error = build(config, llama_tensors()).unwrap_err();
        assert!(
            matches!(error, CompilerError::UnsupportedArchitecture { .. }),
            "removing {key} must be unsupported, got {error:?}"
        );
        assert!(
            error.to_string().contains(key),
            "the error for {key} must name the missing field: {error}"
        );
    }
}

#[test]
fn rejects_a_config_without_any_dimensions() {
    // The shape that previously passed: a bare `model_type` claimed the model
    // and every schema check was skipped.
    let error = build(json!({"model_type": "llama"}), llama_tensors()).unwrap_err();
    assert!(matches!(
        error,
        CompilerError::UnsupportedArchitecture { .. }
    ));
}

#[test]
fn rejects_invalid_attention_dimensions() {
    for (key, value) in [("num_attention_heads", 0), ("num_key_value_heads", 4)] {
        let mut config = llama_config();
        config[key] = json!(value);
        let error = build(config, llama_tensors()).unwrap_err();
        assert!(matches!(
            error,
            CompilerError::UnsupportedArchitecture { .. }
        ));
    }

    // 6 is not divisible by 4 heads, so no integral head_dim exists.
    let mut config = llama_config();
    config["hidden_size"] = json!(6);
    config["num_attention_heads"] = json!(4);
    let error = build(config, llama_tensors()).unwrap_err();
    assert!(matches!(
        error,
        CompilerError::UnsupportedArchitecture { .. }
    ));
}

#[test]
fn rejects_missing_and_misshaped_layer_weights() {
    let mut missing = llama_tensors();
    missing.retain(|tensor| tensor.name.as_str() != "model.layers.0.self_attn.q_proj.weight");
    let error = build(llama_config(), missing).unwrap_err();
    assert!(matches!(error, CompilerError::MissingTensor { .. }));

    let mut misshaped = llama_tensors();
    for tensor in &mut misshaped {
        if tensor.name.as_str() == "model.layers.0.mlp.down_proj.weight" {
            tensor.shape = TensorShape::new(vec![8, 4]);
        }
    }
    let error = build(llama_config(), misshaped).unwrap_err();
    assert!(matches!(error, CompilerError::InvalidTensor { .. }));
}

#[test]
fn untied_models_require_an_explicit_head() {
    let mut config = llama_config();
    config["tie_word_embeddings"] = json!(false);
    let error = build(config.clone(), llama_tensors()).unwrap_err();
    assert!(matches!(error, CompilerError::MissingTensor { .. }));

    let mut tensors = llama_tensors();
    tensors.push(tensor("lm_head.weight", &[4, 4]));
    build(config, tensors).unwrap();
}

fn detect_error(registry: &Registry, artifact: &MockArtifact) -> CompilerError {
    match registry.detect(artifact) {
        Ok(_) => panic!("expected detection to fail"),
        Err(error) => error,
    }
}

#[test]
fn registry_reports_unsupported_ambiguous_and_single_match() {
    let registry = Registry::new();
    let unknown = MockArtifact::new(json!({"model_type": "gpt2"}), llama_tensors());
    assert!(matches!(
        detect_error(&registry, &unknown),
        CompilerError::UnsupportedArchitecture { .. }
    ));

    let llama = MockArtifact::new(llama_config(), llama_tensors());
    assert_eq!(registry.detect(&llama).unwrap().id(), "llama");

    let doubled = Registry::with_adapters(vec![
        Box::new(LlamaDenseAdapter),
        Box::new(LlamaDenseAdapter),
    ]);
    assert!(matches!(
        detect_error(&doubled, &llama),
        CompilerError::AmbiguousArchitecture { .. }
    ));
}
