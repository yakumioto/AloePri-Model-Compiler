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
        let hidden = required_u64(config, "hidden_size")?;
        let layers = required_u64(config, "num_hidden_layers")?;
        let heads = required_u64(config, "num_attention_heads")?;
        let intermediate = required_u64(config, "intermediate_size")?;
        let vocab = required_u64(config, "vocab_size")?;
        // `num_key_value_heads` defaults to `num_attention_heads` and
        // `head_dim` defaults to `hidden_size / num_attention_heads`; those two
        // fallbacks are the ones the Llama config schema defines.
        let kv_heads = optional_u64(config, "num_key_value_heads").unwrap_or(heads);
        let head_dim = optional_u64(config, "head_dim")
            .or_else(|| hidden.checked_div(heads))
            .unwrap_or(0);

        if heads == 0 || kv_heads == 0 || kv_heads > heads || hidden % heads != 0 {
            return Err(CompilerError::UnsupportedArchitecture {
                reason: "invalid Llama attention dimensions".into(),
            });
        }
        let query_dim = heads
            .checked_mul(head_dim)
            .filter(|value| *value == hidden)
            .ok_or_else(|| CompilerError::UnsupportedArchitecture {
                reason: "attention heads and head_dim do not multiply to hidden_size".into(),
            })?;
        let key_value_dim = kv_heads * head_dim;

        let q = [query_dim, hidden];
        let kv = [key_value_dim, hidden];
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
                &[hidden, query_dim],
            )?;
            require_shape(
                artifact,
                &format!("model.layers.{index}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?;
            require_shape(
                artifact,
                &format!("model.layers.{index}.mlp.up_proj.weight"),
                &[intermediate, hidden],
            )?;
            require_shape(
                artifact,
                &format!("model.layers.{index}.mlp.down_proj.weight"),
                &[hidden, intermediate],
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
        require_shape(artifact, "model.norm.weight", &[hidden])?;
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

fn required_u64(config: &Value, key: &str) -> Result<u64> {
    optional_u64(config, key).ok_or_else(|| CompilerError::UnsupportedArchitecture {
        reason: format!("llama config is missing the required `{key}` field"),
    })
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
