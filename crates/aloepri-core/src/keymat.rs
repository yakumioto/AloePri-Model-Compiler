use crate::{CompilerError, ModelFingerprint, Result};
use serde::{Deserialize, Serialize};

pub const KEYMAT_SCHEMA_VERSION: u32 = 4;
pub const KEYMAT_ALGORITHM: &str = "algorithm1-v1";
pub const KEYMAT_BALANCED_ALGORITHM: &str = "algorithm1-balanced-null-v2";

pub fn validate_algorithm(algorithm: &str) -> Result<()> {
    match algorithm {
        KEYMAT_ALGORITHM | KEYMAT_BALANCED_ALGORITHM => Ok(()),
        _ => Err(CompilerError::Unsupported(format!(
            "unsupported KeyMat algorithm {algorithm}"
        ))),
    }
}
pub const KEYMAT_RNG: &str = "chacha20-rand0.9-normal0.5-v1";
pub const KEYMAT_TOLERANCE: f64 = 1e-5;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyMatBinding {
    pub secret_id: String,
    pub source_fingerprint: ModelFingerprint,
    pub hidden_size: u64,
    pub expansion_size: u64,
    pub physical_hidden_size: u64,
    pub lambda_bits: u64,
    pub algorithm: String,
    pub rng: String,
}

impl KeyMatBinding {
    pub fn validate(&self) -> Result<()> {
        validate_algorithm(&self.algorithm)?;
        if physical_hidden_size(self.hidden_size, self.expansion_size)? != self.physical_hidden_size
            || self.rng != KEYMAT_RNG
            || self.secret_id.len() != 64
            || !self
                .secret_id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || lambda_bits(f64::from_bits(self.lambda_bits))? != self.lambda_bits
        {
            return Err(CompilerError::InvalidPlan {
                reason: "invalid KeyMat binding".into(),
            });
        }
        Ok(())
    }
}

pub fn lambda_bits(lambda: f64) -> Result<u64> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(CompilerError::InvalidPlan {
            reason: "KeyMat lambda must be finite and non-negative".into(),
        });
    }
    Ok(if lambda == 0.0 {
        0.0_f64.to_bits()
    } else {
        lambda.to_bits()
    })
}

pub fn physical_hidden_size(d: u64, h: u64) -> Result<u64> {
    if d == 0 || h == 0 || !h.is_multiple_of(2) {
        return Err(CompilerError::InvalidPlan {
            reason: "KeyMat requires d>0 and positive even h".into(),
        });
    }
    d.checked_add(h.checked_mul(2).ok_or(overflow())?)
        .ok_or(overflow())
}

pub fn method_state_bytes(d: u64, h: u64) -> Result<u64> {
    let big_d = physical_hidden_size(d, h)?;
    let bytes = d
        .checked_mul(big_d)
        .and_then(|n| n.checked_mul(16))
        .ok_or(overflow())?;
    usize::try_from(bytes).map_err(|_| overflow())?;
    Ok(bytes)
}

pub fn generation_peak_bytes(d: u64, h: u64) -> Result<u64> {
    let big_d = physical_hidden_size(d, h)?;
    // Covers live bases, full QR/SVD workspaces, products and row-major keys.
    let scratch = big_d
        .checked_mul(big_d)
        .and_then(|n| n.checked_mul(512))
        .ok_or(overflow())?;
    let bytes = scratch
        .checked_add(method_state_bytes(d, h)?)
        .and_then(|n| n.checked_add(128 * 1024))
        .ok_or(overflow())?;
    usize::try_from(bytes).map_err(|_| overflow())?;
    Ok(bytes)
}

pub type TensorSpec = (Vec<u64>, Vec<u64>, crate::plan::OperationKind);

pub fn tensor_specs(
    dimensions: &std::collections::BTreeMap<String, u64>,
    big_d: u64,
) -> Result<std::collections::BTreeMap<String, TensorSpec>> {
    use crate::plan::{KeyMatRole as Role, OperationKind as Op};
    let get = |name: &str| {
        dimensions
            .get(name)
            .copied()
            .filter(|n| *n > 0)
            .ok_or_else(|| CompilerError::InvalidPlan {
                reason: format!("missing positive logical {name}"),
            })
    };
    let d = get("hidden_size")?;
    let v = get("vocab_size")?;
    let m = get("intermediate_size")?;
    let layers = get("num_hidden_layers")?;
    let heads = get("num_attention_heads")?;
    let kv = dimensions
        .get("num_key_value_heads")
        .copied()
        .unwrap_or(heads);
    let head_dim = dimensions.get("head_dim").copied().unwrap_or(d / heads);
    if kv == 0 || !heads.is_multiple_of(kv) || heads.checked_mul(head_dim) != Some(d) {
        return Err(CompilerError::InvalidPlan {
            reason: "invalid logical attention dimensions".into(),
        });
    }
    let kv_dim = kv.checked_mul(head_dim).ok_or(overflow())?;
    let mut specs = std::collections::BTreeMap::new();
    specs.insert(
        "model.embed_tokens.weight".into(),
        (
            vec![v, d],
            vec![v, big_d],
            Op::KeyMatRight {
                role: Role::EmbeddingP,
            },
        ),
    );
    specs.insert(
        "lm_head.weight".into(),
        (
            vec![v, d],
            vec![v, big_d],
            Op::KeyMatRight {
                role: Role::HeadQTranspose,
            },
        ),
    );
    specs.insert("model.norm.weight".into(), (vec![d], vec![d], Op::Copy));
    for i in 0..layers {
        for (suffix, out) in [
            ("self_attn.q_proj", d),
            ("self_attn.k_proj", kv_dim),
            ("self_attn.v_proj", kv_dim),
            ("mlp.gate_proj", m),
            ("mlp.up_proj", m),
        ] {
            specs.insert(
                format!("model.layers.{i}.{suffix}.weight"),
                (
                    vec![out, d],
                    vec![out, big_d],
                    Op::KeyMatRight {
                        role: Role::InputQTranspose,
                    },
                ),
            );
        }
        for (suffix, input) in [("self_attn.o_proj", d), ("mlp.down_proj", m)] {
            specs.insert(
                format!("model.layers.{i}.{suffix}.weight"),
                (
                    vec![d, input],
                    vec![big_d, input],
                    Op::KeyMatLeft {
                        role: Role::OutputPTranspose,
                    },
                ),
            );
        }
        for suffix in ["input_layernorm", "post_attention_layernorm"] {
            specs.insert(
                format!("model.layers.{i}.{suffix}.weight"),
                (vec![d], vec![d], Op::Copy),
            );
        }
    }
    Ok(specs)
}

pub fn validate_plan(plan: &crate::TransformPlan) -> Result<()> {
    let bad = || CompilerError::InvalidPlan {
        reason: "KeyMat runtime, binding or operation geometry mismatch".into(),
    };
    let binding = plan.keymat_binding.as_ref().ok_or_else(bad)?;
    binding.validate()?;
    let rt = &plan.runtime_contract;
    let mut expected = rt.logical_dimensions.clone();
    expected.insert("hidden_size".into(), binding.physical_hidden_size);
    if plan.architecture != "llama"
        || rt.id != "aloepri"
        || rt.version != "1"
        || rt.standard_hf_checkpoint
        || rt.norm_mode.as_deref() != Some("exact_covariant")
        || rt.kv_cache_format.as_deref() != Some("standard_projection_v1")
        || rt.expansion_size != Some(binding.expansion_size)
        || rt.logical_dimensions.get("hidden_size") != Some(&binding.hidden_size)
        || rt.physical_dimensions != expected
        || binding.source_fingerprint != plan.source_fingerprint
        || plan.secret_id.as_ref() != Some(&binding.secret_id)
    {
        return Err(bad());
    }
    let count = rt
        .logical_dimensions
        .get("num_hidden_layers")
        .and_then(|n| n.checked_mul(9))
        .and_then(|n| n.checked_add(3));
    if count != Some(plan.output_inventory.len() as u64)
        || plan.operations.len() != plan.output_inventory.len()
    {
        return Err(bad());
    }
    let specs = tensor_specs(&rt.logical_dimensions, binding.physical_hidden_size)?;
    if specs.len() != plan.operations.len() || specs.len() != plan.output_inventory.len() {
        return Err(bad());
    }
    let head_exists = plan
        .source_inventory
        .iter()
        .any(|t| t.name.as_str() == "lm_head.weight");
    for source in &plan.source_inventory {
        if !specs.contains_key(source.name.as_str()) || source.dtype != crate::DType::F32 {
            return Err(bad());
        }
    }
    for op in &plan.operations {
        let out = &op.output.descriptor;
        let (logical, physical, kind) = specs.get(out.name.as_str()).ok_or_else(bad)?;
        if op.inputs.len() != 1 {
            return Err(bad());
        }
        let input = &op.inputs[0].descriptor;
        let alias = out.name.as_str() == "lm_head.weight"
            && !head_exists
            && input.name.as_str() == "model.embed_tokens.weight";
        if (!alias && input.name != out.name)
            || input.dtype != crate::DType::F32
            || out.dtype != crate::DType::F32
            || input.shape.as_slice() != logical
            || out.shape.as_slice() != physical
            || op.kind != *kind
        {
            return Err(bad());
        }
    }
    let memory = &plan.memory_estimate;
    let minimum = memory
        .metadata_bytes
        .0
        .checked_add(method_state_bytes(
            binding.hidden_size,
            binding.expansion_size,
        )?)
        .and_then(|n| n.checked_add(16))
        .ok_or(overflow())?;
    if memory.method_state_bytes.0
        != method_state_bytes(binding.hidden_size, binding.expansion_size)?
        || memory.peak_bytes.0 < minimum
    {
        return Err(bad());
    }
    Ok(())
}

fn overflow() -> CompilerError {
    CompilerError::ArithmeticOverflow {
        operation: "KeyMat dimensions and memory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dimensions_and_budget_are_checked() {
        assert_eq!(physical_hidden_size(576, 32).unwrap(), 640);
        assert_eq!(method_state_bytes(576, 32).unwrap(), 16 * 576 * 640);
        assert!(generation_peak_bytes(576, 32).unwrap() > method_state_bytes(576, 32).unwrap());
        for (d, h) in [(0, 2), (2, 0), (2, 3), (u64::MAX, 2)] {
            assert!(physical_hidden_size(d, h).is_err());
        }
        for lambda in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(lambda_bits(lambda).is_err());
        }
        assert_eq!(lambda_bits(-0.0).unwrap(), lambda_bits(0.0).unwrap());
    }
}
