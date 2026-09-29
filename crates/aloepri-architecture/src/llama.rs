use aloepri_core::{
    error::{CompilerError, Result},
    model::{ArchitectureAdapter, ModelArtifact},
    plan::{Operation, OperationKind, PlanDraft, TokenRole, TransformConfig},
    types::{ByteLength, ByteOffset, DType, OutputDType, TensorName},
};
use serde_json::Value;

pub struct LlamaDenseAdapter;

impl LlamaDenseAdapter {
    pub fn matches(artifact: &dyn ModelArtifact) -> bool {
        artifact.config().get("model_type").and_then(Value::as_str) == Some("llama")
    }

    fn validate_schema(artifact: &dyn ModelArtifact, token_method: bool) -> Result<()> {
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
        let lm_head = TensorName::try_from("lm_head.weight")?;
        let has_lm_head = artifact.model_spec().tensor(&lm_head).is_some();
        if !tied || has_lm_head {
            require_shape(artifact, "lm_head.weight", &[vocab, hidden])?;
        }
        if token_method {
            if artifact
                .model_spec()
                .tensor(&TensorName::try_from("lm_head.bias")?)
                .is_some()
            {
                return Err(CompilerError::Unsupported(
                    "aloepri-token does not support lm_head.bias".into(),
                ));
            }
            for name in ["model.embed_tokens.weight", "lm_head.weight"] {
                if let Some(tensor) = artifact.model_spec().tensor(&TensorName::try_from(name)?) {
                    if !matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16) {
                        return Err(CompilerError::Unsupported(format!(
                            "aloepri-token does not support {name} dtype {:?}",
                            tensor.dtype
                        )));
                    }
                    if tensor.shape.as_slice().first().copied() != Some(vocab) {
                        return Err(CompilerError::InvalidTensor {
                            name: name.into(),
                            reason: "vocabulary dimension does not match config".into(),
                        });
                    }
                }
            }
            if tied && has_lm_head {
                ensure_tied_weights_match(artifact, &lm_head)?;
            }
        }
        Ok(())
    }
}

fn ensure_tied_weights_match(artifact: &dyn ModelArtifact, lm_head: &TensorName) -> Result<()> {
    let embedding_name = TensorName::try_from("model.embed_tokens.weight")?;
    let embedding = artifact
        .model_spec()
        .tensor(&embedding_name)
        .ok_or_else(|| CompilerError::MissingTensor {
            name: embedding_name.to_string(),
        })?;
    let head =
        artifact
            .model_spec()
            .tensor(lm_head)
            .ok_or_else(|| CompilerError::MissingTensor {
                name: lm_head.to_string(),
            })?;
    if embedding.dtype != head.dtype
        || embedding.shape != head.shape
        || embedding.byte_length != head.byte_length
    {
        return Err(CompilerError::InvalidTensor {
            name: lm_head.to_string(),
            reason: "tied embedding and output projection metadata differ".into(),
        });
    }
    let mut embedding_reader = artifact.tensor_reader(&embedding_name)?;
    let mut head_reader = artifact.tensor_reader(lm_head)?;
    let mut left = vec![0_u8; 64 * 1024];
    let mut right = vec![0_u8; left.len()];
    let mut offset = 0_u64;
    while offset < embedding.byte_length.0 {
        let size = (embedding.byte_length.0 - offset).min(left.len() as u64) as usize;
        embedding_reader.read_bytes(ByteOffset(offset), &mut left[..size])?;
        head_reader.read_bytes(ByteOffset(offset), &mut right[..size])?;
        if left[..size] != right[..size] {
            return Err(CompilerError::InvalidTensor {
                name: lm_head.to_string(),
                reason: "tied embedding and output projection bytes differ".into(),
            });
        }
        offset += size as u64;
    }
    Ok(())
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
        let identity = config.method == aloepri_core::MethodContract::identity();
        let token = config.method == aloepri_core::MethodContract::aloepri_token();
        if !identity && !token {
            return Err(CompilerError::Unsupported(format!(
                "unsupported method {}/{}",
                config.method.id, config.method.version
            )));
        }
        Self::validate_schema(artifact, token)?;
        let operations = artifact
            .tensors()
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                let kind = if token && tensor.name.as_str() == "model.embed_tokens.weight" {
                    OperationKind::TokenPermutation {
                        role: TokenRole::InputEmbedding,
                    }
                } else if token && tensor.name.as_str() == "lm_head.weight" {
                    OperationKind::TokenPermutation {
                        role: TokenRole::OutputProjection,
                    }
                } else {
                    OperationKind::Copy
                };
                Operation {
                    id: aloepri_core::types::OperationId(index as u32),
                    tensor: tensor.name.clone(),
                    source: tensor.clone(),
                    output_dtype: OutputDType::Preserve,
                    memory_requirement: ByteLength(bounded_chunk(tensor.byte_length.0)),
                    dependencies: Vec::new(),
                    kind,
                }
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
