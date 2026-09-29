use crate::{
    error::{CompilerError, Result},
    types::{
        ByteLength, ByteOffset, DType, ModelFingerprint, OperationId, OutputDType,
        TensorDescriptor, TensorName, TensorShape,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MethodContract {
    pub id: String,
    pub version: String,
}

impl MethodContract {
    pub fn identity() -> Self {
        Self {
            id: "identity".into(),
            version: "0.1".into(),
        }
    }

    pub fn aloepri_token() -> Self {
        Self {
            id: "aloepri-token".into(),
            version: "0.1".into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SecretBinding {
    pub secret_id: String,
    pub source_fingerprint: ModelFingerprint,
    pub vocab_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransformConfig {
    pub method: MethodContract,
    pub output_dtype: OutputDType,
    pub memory_limit: ByteLength,
    pub max_shard_size: ByteLength,
    pub workers: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_binding: Option<SecretBinding>,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            method: MethodContract::identity(),
            output_dtype: OutputDType::Preserve,
            memory_limit: ByteLength(256 * 1024 * 1024),
            max_shard_size: ByteLength(4 * 1024 * 1024 * 1024),
            workers: 1,
            secret_binding: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlanDraft {
    pub architecture: String,
    pub operations: Vec<Operation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TokenRole {
    InputEmbedding,
    OutputProjection,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum OperationKind {
    #[default]
    #[serde(rename = "copy")]
    Copy,
    #[serde(rename = "token_permutation")]
    TokenPermutation { role: TokenRole },
}

impl OperationKind {
    pub fn is_copy(&self) -> bool {
        matches!(self, Self::Copy)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub id: OperationId,
    pub tensor: TensorName,
    pub source: TensorDescriptor,
    pub output_dtype: OutputDType,
    pub memory_requirement: ByteLength,
    pub dependencies: Vec<OperationId>,
    #[serde(default, skip_serializing_if = "OperationKind::is_copy")]
    pub kind: OperationKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputTensor {
    pub name: TensorName,
    pub shape: TensorShape,
    pub dtype: DType,
    pub byte_length: ByteLength,
    pub shard: u32,
    pub offset: ByteOffset,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputShard {
    pub id: u32,
    pub filename: String,
    pub payload_length: ByteLength,
    pub file_length: ByteLength,
    pub header: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputLayout {
    pub shards: Vec<OutputShard>,
    pub tensors: Vec<OutputTensor>,
    pub index: Option<Vec<u8>>,
}

impl OutputLayout {
    pub fn tensor(&self, name: &TensorName) -> Option<&OutputTensor> {
        self.tensors.iter().find(|tensor| &tensor.name == name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub metadata_bytes: ByteLength,
    pub io_buffer_bytes: ByteLength,
    pub peak_bytes: ByteLength,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransformPlan {
    pub version: u32,
    pub source_fingerprint: ModelFingerprint,
    pub source_inventory: Vec<TensorDescriptor>,
    pub method: MethodContract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_id: Option<String>,
    pub architecture: String,
    pub operations: Vec<Operation>,
    pub output_layout: OutputLayout,
    pub memory_estimate: MemoryEstimate,
    pub plan_hash: ModelFingerprint,
}

impl TransformPlan {
    pub fn from_draft(
        draft: PlanDraft,
        source_fingerprint: ModelFingerprint,
        source_inventory: Vec<TensorDescriptor>,
        output_layout: OutputLayout,
        config: &TransformConfig,
    ) -> Result<Self> {
        if config.workers != 1 {
            return Err(CompilerError::Unsupported(
                "v0.1 uses a single worker".into(),
            ));
        }
        if config.output_dtype != OutputDType::Preserve {
            return Err(CompilerError::Unsupported(
                "the supported methods preserve the source dtype".into(),
            ));
        }
        let is_identity = config.method == MethodContract::identity();
        let is_token = config.method == MethodContract::aloepri_token();
        if !is_identity && !is_token {
            return Err(CompilerError::Unsupported(format!(
                "unsupported method {}/{}",
                config.method.id, config.method.version
            )));
        }
        let secret_id = if is_token {
            let binding =
                config
                    .secret_binding
                    .as_ref()
                    .ok_or_else(|| CompilerError::InvalidPlan {
                        reason: "aloepri-token requires a client secret binding".into(),
                    })?;
            if binding.source_fingerprint != source_fingerprint {
                return Err(CompilerError::InvalidPlan {
                    reason: "client secret source fingerprint does not match the input".into(),
                });
            }
            if binding.vocab_size < 2 {
                return Err(CompilerError::InvalidPlan {
                    reason: "client secret vocabulary must contain at least two tokens".into(),
                });
            }
            Some(binding.secret_id.clone())
        } else {
            if config.secret_binding.is_some() {
                return Err(CompilerError::InvalidPlan {
                    reason: "identity does not accept a client secret binding".into(),
                });
            }
            None
        };
        if is_identity
            && draft
                .operations
                .iter()
                .any(|operation| !operation.kind.is_copy())
        {
            return Err(CompilerError::InvalidPlan {
                reason: "identity operations must be byte copies".into(),
            });
        }
        let base_metadata = (source_inventory.len() as u64).checked_mul(256).ok_or(
            CompilerError::ArithmeticOverflow {
                operation: "plan metadata estimate",
            },
        )?;
        let mapping_bytes = config
            .secret_binding
            .as_ref()
            .map(|binding| {
                binding
                    .vocab_size
                    .checked_mul(8)
                    .ok_or(CompilerError::ArithmeticOverflow {
                        operation: "token permutation memory estimate",
                    })
            })
            .transpose()?
            .unwrap_or(0);
        let metadata_bytes =
            base_metadata
                .checked_add(mapping_bytes)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "plan metadata estimate",
                })?;
        if config.memory_limit.0 <= metadata_bytes {
            return Err(CompilerError::MemoryLimitExceeded {
                requested: metadata_bytes.saturating_add(1),
                available: config.memory_limit.0,
            });
        }
        let io_buffer_bytes = (config.memory_limit.0 - metadata_bytes).min(4 * 1024 * 1024);
        let mut plan = Self {
            version: if is_token { 2 } else { 1 },
            source_fingerprint,
            source_inventory,
            method: config.method.clone(),
            secret_id,
            architecture: draft.architecture,
            operations: draft.operations,
            output_layout,
            memory_estimate: MemoryEstimate {
                metadata_bytes: ByteLength(metadata_bytes),
                io_buffer_bytes: ByteLength(io_buffer_bytes),
                peak_bytes: ByteLength(metadata_bytes + io_buffer_bytes),
            },
            plan_hash: ModelFingerprint::from_digest(blake3::hash(b"uninitialized")),
        };
        plan.validate()?;
        plan.plan_hash = plan.compute_hash()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        let identity = self.method == MethodContract::identity();
        let token = self.method == MethodContract::aloepri_token();
        if (!identity && !token) || (identity && self.version != 1) || (token && self.version != 2)
        {
            return Err(CompilerError::UnsupportedVersion {
                version: self.version,
            });
        }
        if (identity && self.secret_id.is_some()) || (token && self.secret_id.is_none()) {
            return Err(CompilerError::InvalidPlan {
                reason: "method and secret binding metadata do not agree".into(),
            });
        }
        let source_names: BTreeSet<_> = self
            .source_inventory
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        let operation_names: BTreeSet<_> = self
            .operations
            .iter()
            .map(|operation| operation.tensor.clone())
            .collect();
        let output_names: BTreeSet<_> = self
            .output_layout
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        if source_names.len() != self.source_inventory.len()
            || operation_names.len() != self.operations.len()
            || output_names.len() != self.output_layout.tensors.len()
        {
            return Err(CompilerError::InvalidPlan {
                reason: "duplicate tensor names in plan".into(),
            });
        }
        if source_names != operation_names || source_names != output_names {
            return Err(CompilerError::InvalidPlan {
                reason: "source, operation, and output inventories differ".into(),
            });
        }
        let operation_ids: BTreeSet<_> = self
            .operations
            .iter()
            .map(|operation| operation.id)
            .collect();
        if operation_ids.len() != self.operations.len() {
            return Err(CompilerError::InvalidPlan {
                reason: "duplicate operation ids in plan".into(),
            });
        }
        let mut ranges_by_shard: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
        for output in &self.output_layout.tensors {
            let shard = self
                .output_layout
                .shards
                .get(output.shard as usize)
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: format!("output {} references an unknown shard", output.name),
                })?;
            let end = output.offset.0.checked_add(output.byte_length.0).ok_or(
                CompilerError::ArithmeticOverflow {
                    operation: "output range validation",
                },
            )?;
            if end > shard.payload_length.0 {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("output {} exceeds its shard payload", output.name),
                });
            }
            ranges_by_shard
                .entry(output.shard)
                .or_default()
                .push((output.offset.0, end));
        }
        for ranges in ranges_by_shard.values_mut() {
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                if pair[1].0 < pair[0].1 {
                    return Err(CompilerError::InvalidPlan {
                        reason: "output ranges overlap".into(),
                    });
                }
            }
        }
        for operation in &self.operations {
            if operation
                .dependencies
                .iter()
                .any(|dependency| !operation_ids.contains(dependency))
            {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("operation {} has an unknown dependency", operation.id.0),
                });
            }
            let output = self
                .output_layout
                .tensor(&operation.tensor)
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: format!("missing output for {}", operation.tensor),
                })?;
            if output.byte_length != operation.source.byte_length
                || output.dtype != operation.source.dtype
                || output.shape != operation.source.shape
            {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("output metadata differs for {}", operation.tensor),
                });
            }
        }
        Ok(())
    }

    pub fn compute_hash(&self) -> Result<ModelFingerprint> {
        let mut canonical = self.clone();
        canonical.plan_hash = ModelFingerprint::from_digest(blake3::hash(b"excluded"));
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| CompilerError::Invariant(error.to_string()))?;
        Ok(ModelFingerprint::from_digest(blake3::hash(&bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ShardId, TensorLocation, TensorShape};

    fn descriptor(name: &str) -> TensorDescriptor {
        TensorDescriptor {
            name: TensorName::try_from(name).unwrap(),
            shape: TensorShape::new(vec![2]),
            dtype: DType::U8,
            byte_length: ByteLength(2),
            location: TensorLocation {
                shard: ShardId(0),
                offset: ByteOffset(0),
                length: ByteLength(2),
            },
        }
    }

    #[test]
    fn plan_hash_is_deterministic() {
        let source = descriptor("a");
        let output = OutputLayout {
            shards: vec![OutputShard {
                id: 0,
                filename: "model.safetensors".into(),
                payload_length: ByteLength(2),
                file_length: ByteLength(10),
                header: vec![],
            }],
            tensors: vec![OutputTensor {
                name: source.name.clone(),
                shape: source.shape.clone(),
                dtype: source.dtype,
                byte_length: source.byte_length,
                shard: 0,
                offset: ByteOffset(0),
            }],
            index: None,
        };
        let draft = PlanDraft {
            architecture: "test".into(),
            operations: vec![Operation {
                id: OperationId(0),
                tensor: source.name.clone(),
                source: source.clone(),
                output_dtype: OutputDType::Preserve,
                memory_requirement: ByteLength(1),
                dependencies: vec![],
                kind: OperationKind::Copy,
            }],
        };
        let first = TransformPlan::from_draft(
            draft.clone(),
            ModelFingerprint::from_digest(blake3::hash(b"source")),
            vec![source.clone()],
            output.clone(),
            &TransformConfig::default(),
        )
        .unwrap();
        let second = TransformPlan::from_draft(
            draft,
            ModelFingerprint::from_digest(blake3::hash(b"source")),
            vec![source],
            output,
            &TransformConfig::default(),
        )
        .unwrap();
        assert_eq!(first.plan_hash, second.plan_hash);
    }
}
