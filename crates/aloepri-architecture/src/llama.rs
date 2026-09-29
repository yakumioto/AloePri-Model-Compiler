use aloepri_core::{
    error::{CompilerError, Result},
    model::{ArchitectureAdapter, ModelArtifact},
    plan::{Operation, PlanDraft, TransformConfig},
    types::{ByteLength, OutputDType, TensorName},
};
use serde_json::Value;

pub struct LlamaDenseAdapter;

impl LlamaDenseAdapter {
    pub fn matches(artifact: &dyn ModelArtifact) -> bool {
        artifact.config().get("model_type").and_then(Value::as_str) == Some("llama")
    }

    fn validate_schema(artifact: &dyn ModelArtifact) -> Result<()> {
        let config = artifact.config();
        let hidden = optional_u64(config, "hidden_size");
        let layers = optional_u64(config, "num_hidden_layers");
        let heads = optional_u64(config, "num_attention_heads");
        let kv_heads = optional_u64(config, "num_key_value_heads").or(heads);
        let intermediate = optional_u64(config, "intermediate_size");
        let vocab = optional_u64(config, "vocab_size");
        let head_dim = optional_u64(config, "head_dim").or_else(|| {
            hidden
                .zip(heads)
                .and_then(|(hidden, heads)| (heads != 0).then_some(hidden / heads))
        });
        if let (Some(hidden), Some(heads), Some(kv_heads), Some(head_dim)) =
            (hidden, heads, kv_heads, head_dim)
        {
            if heads == 0
                || kv_heads == 0
                || kv_heads > heads
                || heads.checked_mul(head_dim) != Some(hidden)
            {
                return Err(CompilerError::UnsupportedArchitecture {
                    reason: "invalid Llama attention dimensions".into(),
                });
            }
            let q = [heads * head_dim, hidden];
            let kv = [kv_heads * head_dim, hidden];
            if let Some(layers) = layers {
                for index in 0..layers {
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.self_attn.q_proj.weight"),
                        &q,
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.self_attn.k_proj.weight"),
                        &kv,
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.self_attn.v_proj.weight"),
                        &kv,
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.self_attn.o_proj.weight"),
                        &[hidden, heads * head_dim],
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.mlp.gate_proj.weight"),
                        &[intermediate.unwrap_or(0), hidden],
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.mlp.up_proj.weight"),
                        &[intermediate.unwrap_or(0), hidden],
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.mlp.down_proj.weight"),
                        &[hidden, intermediate.unwrap_or(0)],
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.input_layernorm.weight"),
                        &[hidden],
                    )?;
                    require_shape(
                        artifact,
                        &format!("model.layers.{index}.post_attention_layernorm.weight"),
                        &[hidden],
                    )?;
                }
            }
            require_shape(artifact, "model.norm.weight", &[hidden])?;
            if let Some(vocab) = vocab {
                require_shape(artifact, "model.embed_tokens.weight", &[vocab, hidden])?;
                let tied = config
                    .get("tie_word_embeddings")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if !tied
                    || artifact
                        .model_spec()
                        .tensor(&TensorName::try_from("lm_head.weight")?)
                        .is_some()
                {
                    require_shape(artifact, "lm_head.weight", &[vocab, hidden])?;
                }
            }
        }
        Ok(())
    }
}

impl ArchitectureAdapter for LlamaDenseAdapter {
    fn id(&self) -> &'static str {
        "llama"
    }

    fn build_plan(
        &self,
        artifact: &dyn ModelArtifact,
        config: &TransformConfig,
    ) -> Result<PlanDraft> {
        if config.method.id != "identity" {
            return Err(CompilerError::Unsupported(
                "only identity is available".into(),
            ));
        }
        Self::validate_schema(artifact)?;
        let operations = artifact
            .tensors()
            .iter()
            .enumerate()
            .map(|(index, tensor)| Operation {
                id: aloepri_core::types::OperationId(index as u32),
                tensor: tensor.name.clone(),
                source: tensor.clone(),
                output_dtype: OutputDType::Preserve,
                memory_requirement: ByteLength(bounded_chunk(tensor.byte_length.0)),
                dependencies: Vec::new(),
            })
            .collect();
        Ok(PlanDraft {
            architecture: "llama".into(),
            operations,
        })
    }
}

fn bounded_chunk(length: u64) -> u64 {
    if length == 0 {
        1
    } else {
        length.min(4 * 1024 * 1024)
    }
}

fn optional_u64(config: &Value, key: &str) -> Option<u64> {
    config.get(key).and_then(Value::as_u64)
}

fn require_shape(artifact: &dyn ModelArtifact, name: &str, expected: &[u64]) -> Result<()> {
    let name = TensorName::try_from(name)?;
    let tensor =
        artifact
            .model_spec()
            .tensor(&name)
            .ok_or_else(|| CompilerError::MissingTensor {
                name: name.to_string(),
            })?;
    if tensor.shape.as_slice() != expected {
        return Err(CompilerError::InvalidTensor {
            name: name.to_string(),
            reason: format!("expected shape {expected:?}, got {}", tensor.shape),
        });
    }
    Ok(())
}
