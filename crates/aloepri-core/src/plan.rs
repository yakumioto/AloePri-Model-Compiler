use crate::{
    error::{CompilerError, Result},
    types::{
        ByteLength, ByteOffset, DType, ModelFingerprint, OperationId, OutputDType,
        OutputTensorDescriptor, TensorDescriptor, TensorName, TensorShape,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Serialized schema version shared by `TransformPlan`, manifest and checkpoint.
///
/// v1/v2 artifacts stay readable by their own legacy verification branch, but a
/// v1/v2 checkpoint never resumes a v3 execution and vice versa.
pub const SCHEMA_VERSION: u32 = 3;

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

    pub fn aloepri_keymat() -> Self {
        Self {
            id: "aloepri-keymat".into(),
            version: "0.1".into(),
        }
    }

    pub fn aloepri_token() -> Self {
        Self {
            id: "aloepri-token".into(),
            version: "0.1".into(),
        }
    }

    /// A diagnostic shape-changing method. It exists to exercise the
    /// shape-independent pipeline and is never exposed by the product CLI.
    pub fn expand_test() -> Self {
        Self {
            id: "expand-test".into(),
            version: "0.1".into(),
        }
    }
}

/// The runtime an artifact targets, separated from the logical model config.
///
/// `config.json` keeps describing the logical architecture; this contract
/// records whether the published tensors are a standard HF checkpoint and what
/// physical dimensions they actually carry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeContract {
    pub id: String,
    pub version: String,
    pub standard_hf_checkpoint: bool,
    pub architecture: String,
    pub logical_dimensions: BTreeMap<String, u64>,
    pub physical_dimensions: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expansion_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_format: Option<String>,
}

impl RuntimeContract {
    pub fn huggingface(
        architecture: impl Into<String>,
        logical_dimensions: BTreeMap<String, u64>,
    ) -> Self {
        Self {
            id: "huggingface".into(),
            version: "1".into(),
            standard_hf_checkpoint: true,
            architecture: architecture.into(),
            physical_dimensions: logical_dimensions.clone(),
            logical_dimensions,
            expansion_size: None,
            norm_mode: None,
            kv_cache_format: None,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keymat_binding: Option<crate::keymat::KeyMatBinding>,
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
            keymat_binding: None,
        }
    }
}

/// The adapter's proposal: a runtime contract plus one operation per output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlanDraft {
    pub architecture: String,
    pub runtime_contract: RuntimeContract,
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
    /// Row-wise padding: each output row is the source row followed by
    /// `additional_columns` zero elements. Diagnostic method only.
    #[serde(rename = "pad_columns")]
    PadColumns { additional_columns: u64 },
    #[serde(rename = "keymat_right")]
    KeyMatRight { role: KeyMatRole },
    #[serde(rename = "keymat_left")]
    KeyMatLeft { role: KeyMatRole },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum KeyMatRole {
    EmbeddingP,
    InputQTranspose,
    HeadQTranspose,
    OutputPTranspose,
}

impl OperationKind {
    pub fn is_copy(&self) -> bool {
        matches!(self, Self::Copy)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperationInput {
    pub descriptor: TensorDescriptor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperationOutput {
    pub descriptor: OutputTensorDescriptor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub id: OperationId,
    #[serde(default, skip_serializing_if = "OperationKind::is_copy")]
    pub kind: OperationKind,
    pub inputs: Vec<OperationInput>,
    pub output: OperationOutput,
    pub memory_requirement: ByteLength,
    pub dependencies: Vec<OperationId>,
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

impl From<&OutputTensorDescriptor> for OutputTensor {
    fn from(descriptor: &OutputTensorDescriptor) -> Self {
        Self {
            name: descriptor.name.clone(),
            shape: descriptor.shape.clone(),
            dtype: descriptor.dtype,
            byte_length: descriptor.byte_length,
            shard: 0,
            offset: ByteOffset(0),
        }
    }
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

/// Layered memory accounting for the execution phase.
///
/// `metadata_bytes` is an estimate of the resident plan/inventory metadata; the
/// buffer fields describe the transient working set of a single operation when
/// `workers == 1`. None of these claim to bound process RSS.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub metadata_bytes: ByteLength,
    pub input_buffer_bytes: ByteLength,
    pub output_buffer_bytes: ByteLength,
    pub transform_scratch_bytes: ByteLength,
    pub method_state_bytes: ByteLength,
    pub peak_bytes: ByteLength,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransformPlan {
    pub version: u32,
    pub source_fingerprint: ModelFingerprint,
    pub source_inventory: Vec<TensorDescriptor>,
    pub output_inventory: Vec<OutputTensorDescriptor>,
    pub method: MethodContract,
    pub runtime_contract: RuntimeContract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keymat_binding: Option<crate::keymat::KeyMatBinding>,
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
        let is_expand = config.method == MethodContract::expand_test();
        let is_keymat = config.method == MethodContract::aloepri_keymat();
        if !is_identity && !is_token && !is_expand && !is_keymat {
            return Err(CompilerError::Unsupported(format!(
                "unsupported method {}/{}",
                config.method.id, config.method.version
            )));
        }
        if !is_keymat && config.keymat_binding.is_some() {
            return Err(CompilerError::InvalidPlan {
                reason: "only KeyMat accepts a KeyMat binding".into(),
            });
        }
        let secret_id = if is_keymat {
            let binding =
                config
                    .keymat_binding
                    .as_ref()
                    .ok_or_else(|| CompilerError::InvalidPlan {
                        reason: "KeyMat requires a secret binding".into(),
                    })?;
            binding.validate()?;
            if binding.source_fingerprint != source_fingerprint || config.secret_binding.is_some() {
                return Err(CompilerError::InvalidPlan {
                    reason: "KeyMat source or secret binding mismatch".into(),
                });
            }
            Some(binding.secret_id.clone())
        } else if is_token {
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
                    reason: "this method does not accept a client secret binding".into(),
                });
            }
            None
        };

        let source_bytes = checked_sum(source_inventory.iter().map(|t| t.byte_length.0))?;
        let output_bytes = checked_sum(
            draft
                .operations
                .iter()
                .map(|operation| operation.output.descriptor.byte_length.0),
        )?;
        let mut output_inventory: Vec<OutputTensorDescriptor> = draft
            .operations
            .iter()
            .map(|operation| operation.output.descriptor.clone())
            .collect();
        output_inventory.sort_by(|left, right| left.name.cmp(&right.name));

        let metadata_entries = source_inventory
            .len()
            .checked_add(output_inventory.len())
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "plan metadata estimate",
            })? as u64;
        let base_metadata =
            metadata_entries
                .checked_mul(256)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "plan metadata estimate",
                })?;
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
        let buffer_bytes = (config.memory_limit.0 - metadata_bytes).min(4 * 1024 * 1024);
        let peak_bytes =
            metadata_bytes
                .checked_add(buffer_bytes)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "plan peak memory estimate",
                })?;

        let mut plan = Self {
            version: if is_keymat {
                crate::keymat::KEYMAT_SCHEMA_VERSION
            } else {
                SCHEMA_VERSION
            },
            keymat_binding: config.keymat_binding.clone(),
            source_fingerprint,
            source_inventory,
            output_inventory,
            method: config.method.clone(),
            runtime_contract: draft.runtime_contract,
            secret_id,
            architecture: draft.architecture,
            operations: draft.operations,
            output_layout,
            memory_estimate: MemoryEstimate {
                metadata_bytes: ByteLength(metadata_bytes),
                input_buffer_bytes: ByteLength(buffer_bytes.min(source_bytes.max(1))),
                output_buffer_bytes: ByteLength(buffer_bytes.min(output_bytes.max(1))),
                transform_scratch_bytes: ByteLength(mapping_bytes),
                method_state_bytes: ByteLength(mapping_bytes),
                peak_bytes: ByteLength(peak_bytes),
            },
            plan_hash: ModelFingerprint::from_digest(blake3::hash(b"uninitialized")),
        };
        if let Some(binding) = &config.keymat_binding {
            let state =
                crate::keymat::method_state_bytes(binding.hidden_size, binding.expansion_size)?;
            let resident =
                metadata_bytes
                    .checked_add(state)
                    .ok_or(CompilerError::ArithmeticOverflow {
                        operation: "KeyMat resident memory",
                    })?;
            let available = config
                .memory_limit
                .0
                .checked_sub(resident)
                .filter(|n| *n >= 16)
                .ok_or(CompilerError::MemoryLimitExceeded {
                    requested: resident.saturating_add(16),
                    available: config.memory_limit.0,
                })?;
            let tile = available.min(4 * 1024 * 1024) / 16;
            plan.memory_estimate = MemoryEstimate {
                metadata_bytes: ByteLength(metadata_bytes),
                method_state_bytes: ByteLength(state),
                input_buffer_bytes: ByteLength(tile * 4),
                output_buffer_bytes: ByteLength(tile * 4),
                transform_scratch_bytes: ByteLength(tile * 8),
                peak_bytes: ByteLength(resident + tile * 16),
            };
        }
        plan.validate()?;
        plan.plan_hash = plan.compute_hash()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        let expected_version = if self.method == MethodContract::aloepri_keymat() {
            crate::keymat::KEYMAT_SCHEMA_VERSION
        } else {
            SCHEMA_VERSION
        };
        if self.version != expected_version {
            return Err(CompilerError::UnsupportedVersion {
                version: self.version,
            });
        }
        self.validate_runtime()?;
        let identity = self.method == MethodContract::identity();
        let token = self.method == MethodContract::aloepri_token();
        if (identity && self.secret_id.is_some()) || (token && self.secret_id.is_none()) {
            return Err(CompilerError::InvalidPlan {
                reason: "method and secret binding metadata do not agree".into(),
            });
        }

        let source_by_name = unique_descriptors(&self.source_inventory, "source")?;
        let output_by_name = unique_outputs(&self.output_inventory, "output")?;
        for descriptor in &self.source_inventory {
            if descriptor.expected_byte_length()? != descriptor.byte_length {
                return Err(CompilerError::InvalidPlan {
                    reason: format!(
                        "source tensor {} byte length disagrees with its shape",
                        descriptor.name
                    ),
                });
            }
        }
        for descriptor in &self.output_inventory {
            if descriptor.expected_byte_length()? != descriptor.byte_length {
                return Err(CompilerError::InvalidPlan {
                    reason: format!(
                        "output tensor {} byte length disagrees with its shape",
                        descriptor.name
                    ),
                });
            }
        }

        let mut operation_ids = BTreeSet::new();
        let mut produced = BTreeSet::new();
        let mut indexes = BTreeMap::new();
        for (index, operation) in self.operations.iter().enumerate() {
            if !operation_ids.insert(operation.id) {
                return Err(CompilerError::InvalidPlan {
                    reason: "duplicate operation ids in plan".into(),
                });
            }
            indexes.insert(operation.id, index);
        }
        for (index, operation) in self.operations.iter().enumerate() {
            if operation.inputs.is_empty() {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("operation {} has no inputs", operation.id.0),
                });
            }
            let mut seen_inputs = BTreeSet::new();
            for input in &operation.inputs {
                if !seen_inputs.insert(input.descriptor.name.clone()) {
                    return Err(CompilerError::InvalidPlan {
                        reason: format!("operation {} lists the same input twice", operation.id.0),
                    });
                }
                let source = source_by_name.get(&input.descriptor.name).ok_or_else(|| {
                    CompilerError::InvalidPlan {
                        reason: format!(
                            "operation {} references missing source {}",
                            operation.id.0, input.descriptor.name
                        ),
                    }
                })?;
                if *source != &input.descriptor {
                    return Err(CompilerError::InvalidPlan {
                        reason: format!(
                            "operation {} input {} disagrees with the source inventory",
                            operation.id.0, input.descriptor.name
                        ),
                    });
                }
            }
            let name = &operation.output.descriptor.name;
            let expected = output_by_name
                .get(name)
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: format!("operation {} has unknown output {}", operation.id.0, name),
                })?;
            if *expected != &operation.output.descriptor {
                return Err(CompilerError::InvalidPlan {
                    reason: format!(
                        "operation {} output {} disagrees with the output inventory",
                        operation.id.0, name
                    ),
                });
            }
            if !produced.insert(name.clone()) {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("output {} has more than one producer", name),
                });
            }
            for dependency in &operation.dependencies {
                match indexes.get(dependency) {
                    Some(position) if *position < index => {}
                    Some(_) => {
                        return Err(CompilerError::InvalidPlan {
                            reason: format!(
                                "operation {} depends on a later operation",
                                operation.id.0
                            ),
                        });
                    }
                    None => {
                        return Err(CompilerError::InvalidPlan {
                            reason: format!(
                                "operation {} has an unknown dependency",
                                operation.id.0
                            ),
                        });
                    }
                }
            }
        }
        if produced.len() != output_by_name.len() {
            return Err(CompilerError::InvalidPlan {
                reason: "not every output inventory entry has a producer".into(),
            });
        }
        self.validate_layout(&output_by_name)?;
        self.validate_method_operations()?;
        Ok(())
    }

    fn validate_runtime(&self) -> Result<()> {
        let runtime = &self.runtime_contract;
        if runtime.architecture != self.architecture {
            return Err(CompilerError::InvalidPlan {
                reason: "runtime contract architecture disagrees with the plan".into(),
            });
        }
        let identity = self.method == MethodContract::identity();
        let token = self.method == MethodContract::aloepri_token();
        let expand = self.method == MethodContract::expand_test();
        let keymat = self.method == MethodContract::aloepri_keymat();
        if !keymat
            && (self.keymat_binding.is_some()
                || runtime.norm_mode.is_some()
                || runtime.kv_cache_format.is_some())
        {
            return Err(CompilerError::InvalidPlan {
                reason: "KeyMat fields require KeyMat method".into(),
            });
        }
        if keymat {
            return crate::keymat::validate_plan(self);
        }
        if !identity && !token && !expand {
            return Err(CompilerError::Unsupported(format!(
                "unsupported method {}/{}",
                self.method.id, self.method.version
            )));
        }
        if identity || token {
            if !runtime.standard_hf_checkpoint
                || runtime.id != "huggingface"
                || runtime.expansion_size.is_some()
            {
                return Err(CompilerError::InvalidPlan {
                    reason: "a standard method must keep the huggingface runtime".into(),
                });
            }
            if runtime.logical_dimensions != runtime.physical_dimensions {
                return Err(CompilerError::InvalidPlan {
                    reason: "a standard runtime must not change physical dimensions".into(),
                });
            }
        } else {
            let expansion = runtime
                .expansion_size
                .filter(|value| *value > 0)
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: "the diagnostic runtime requires a positive expansion size".into(),
                })?;
            if runtime.standard_hf_checkpoint || runtime.id != "expand-test-runtime" {
                return Err(CompilerError::InvalidPlan {
                    reason: "a shape-changing runtime must not claim standard HF".into(),
                });
            }
            let logical_hidden = runtime
                .logical_dimensions
                .get("hidden_size")
                .copied()
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: "the diagnostic runtime needs a logical hidden_size".into(),
                })?;
            let physical_hidden = runtime
                .physical_dimensions
                .get("hidden_size")
                .copied()
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: "the diagnostic runtime needs a physical hidden_size".into(),
                })?;
            let expected = logical_hidden.checked_add(expansion.checked_mul(2).ok_or(
                CompilerError::ArithmeticOverflow {
                    operation: "runtime physical hidden size",
                },
            )?);
            if expected != Some(physical_hidden) {
                return Err(CompilerError::InvalidPlan {
                    reason: "physical hidden size must equal logical + 2 * expansion".into(),
                });
            }
        }
        Ok(())
    }

    fn validate_method_operations(&self) -> Result<()> {
        let identity = self.method == MethodContract::identity();
        let token = self.method == MethodContract::aloepri_token();
        if self.method == MethodContract::aloepri_keymat() {
            return Ok(());
        }
        for operation in &self.operations {
            let input = &operation.inputs[0].descriptor;
            let output = &operation.output.descriptor;
            match operation.kind {
                OperationKind::Copy => {
                    if input.name != output.name
                        || input.shape != output.shape
                        || input.dtype != output.dtype
                        || input.byte_length != output.byte_length
                    {
                        return Err(CompilerError::InvalidPlan {
                            reason: format!(
                                "copy operation {} must keep name, shape and dtype",
                                operation.id.0
                            ),
                        });
                    }
                }
                OperationKind::TokenPermutation { .. } => {
                    if !token {
                        return Err(CompilerError::InvalidPlan {
                            reason: "token permutation requires aloepri-token".into(),
                        });
                    }
                    if input.name != output.name
                        || input.shape != output.shape
                        || input.dtype != output.dtype
                        || input.byte_length != output.byte_length
                    {
                        return Err(CompilerError::InvalidPlan {
                            reason: format!(
                                "token permutation {} must keep name, shape and dtype",
                                operation.id.0
                            ),
                        });
                    }
                }
                OperationKind::KeyMatRight { .. } | OperationKind::KeyMatLeft { .. } => {
                    return Err(CompilerError::InvalidPlan {
                        reason: "KeyMat operation requires KeyMat method".into(),
                    });
                }
                OperationKind::PadColumns { additional_columns } => {
                    if identity || token {
                        return Err(CompilerError::InvalidPlan {
                            reason: "padding requires the diagnostic method".into(),
                        });
                    }
                    let expansion = self.runtime_contract.expansion_size.ok_or_else(|| {
                        CompilerError::InvalidPlan {
                            reason: "padding requires a runtime expansion size".into(),
                        }
                    })?;
                    if additional_columns != expansion * 2 {
                        return Err(CompilerError::InvalidPlan {
                            reason: "padding width must match the runtime expansion".into(),
                        });
                    }
                    if input.dtype != output.dtype || input.shape.as_slice().len() != 2 {
                        return Err(CompilerError::InvalidPlan {
                            reason: "padding requires a two-dimensional tensor of one dtype".into(),
                        });
                    }
                    let rows = input.shape.as_slice()[0];
                    let logical_columns = input.shape.as_slice()[1];
                    let logical_hidden = self
                        .runtime_contract
                        .logical_dimensions
                        .get("hidden_size")
                        .copied()
                        .unwrap_or(u64::MAX);
                    if logical_columns != logical_hidden {
                        return Err(CompilerError::InvalidPlan {
                            reason: "padding requires the logical hidden dimension".into(),
                        });
                    }
                    let padded = logical_columns.checked_add(additional_columns).ok_or(
                        CompilerError::ArithmeticOverflow {
                            operation: "padded column count",
                        },
                    )?;
                    if output.shape.as_slice() != [rows, padded] {
                        return Err(CompilerError::InvalidPlan {
                            reason: "padding output shape does not match its width".into(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_layout(
        &self,
        output_by_name: &BTreeMap<TensorName, &OutputTensorDescriptor>,
    ) -> Result<()> {
        if self.output_layout.tensors.len() != self.output_inventory.len() {
            return Err(CompilerError::InvalidPlan {
                reason: "layout tensor count differs from the output inventory".into(),
            });
        }
        let mut seen = BTreeSet::new();
        let mut ranges_by_shard: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
        for output in &self.output_layout.tensors {
            if !seen.insert(output.name.clone()) {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("layout lists {} twice", output.name),
                });
            }
            let expected =
                output_by_name
                    .get(&output.name)
                    .ok_or_else(|| CompilerError::InvalidPlan {
                        reason: format!("layout contains unknown output {}", output.name),
                    })?;
            if expected.shape != output.shape
                || expected.dtype != output.dtype
                || expected.byte_length != output.byte_length
            {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("layout metadata differs for {}", output.name),
                });
            }
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
        Ok(())
    }

    pub fn compute_hash(&self) -> Result<ModelFingerprint> {
        let mut canonical = self.clone();
        canonical.plan_hash = ModelFingerprint::from_digest(blake3::hash(b"excluded"));
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| CompilerError::Invariant(error.to_string()))?;
        Ok(ModelFingerprint::from_digest(blake3::hash(&bytes)))
    }

    /// Recompute the plan hash and reject a serialized plan whose stored hash
    /// does not describe its contents.
    pub fn verify_hash(&self) -> Result<()> {
        if self.compute_hash()? != self.plan_hash {
            return Err(CompilerError::InvalidPlan {
                reason: "embedded plan hash does not describe the plan".into(),
            });
        }
        Ok(())
    }
}

fn checked_sum(values: impl Iterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |accumulator, value| {
        accumulator
            .checked_add(value)
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "inventory byte total",
            })
    })
}

fn unique_descriptors<'a>(
    descriptors: &'a [TensorDescriptor],
    label: &str,
) -> Result<BTreeMap<TensorName, &'a TensorDescriptor>> {
    let mut map = BTreeMap::new();
    for descriptor in descriptors {
        if map.insert(descriptor.name.clone(), descriptor).is_some() {
            return Err(CompilerError::InvalidPlan {
                reason: format!("duplicate {label} tensor names in plan"),
            });
        }
    }
    Ok(map)
}

fn unique_outputs<'a>(
    descriptors: &'a [OutputTensorDescriptor],
    label: &str,
) -> Result<BTreeMap<TensorName, &'a OutputTensorDescriptor>> {
    let mut map = BTreeMap::new();
    for descriptor in descriptors {
        if map.insert(descriptor.name.clone(), descriptor).is_some() {
            return Err(CompilerError::InvalidPlan {
                reason: format!("duplicate {label} tensor names in plan"),
            });
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ShardId, TensorLocation};

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

    fn layout_for(descriptor: &TensorDescriptor) -> OutputLayout {
        OutputLayout {
            shards: vec![OutputShard {
                id: 0,
                filename: "model.safetensors".into(),
                payload_length: ByteLength(2),
                file_length: ByteLength(10),
                header: vec![],
            }],
            tensors: vec![OutputTensor {
                name: descriptor.name.clone(),
                shape: descriptor.shape.clone(),
                dtype: descriptor.dtype,
                byte_length: descriptor.byte_length,
                shard: 0,
                offset: ByteOffset(0),
            }],
            index: None,
        }
    }

    fn copy_operation(id: u32, source: &TensorDescriptor) -> Operation {
        Operation {
            id: OperationId(id),
            kind: OperationKind::Copy,
            inputs: vec![OperationInput {
                descriptor: source.clone(),
            }],
            output: OperationOutput {
                descriptor: source.into(),
            },
            memory_requirement: ByteLength(1),
            dependencies: vec![],
        }
    }

    fn standard_draft(operations: Vec<Operation>) -> PlanDraft {
        PlanDraft {
            architecture: "test".into(),
            runtime_contract: RuntimeContract::huggingface("test", BTreeMap::new()),
            operations,
        }
    }

    #[test]
    fn plan_hash_is_deterministic() {
        let source = descriptor("a");
        let output = layout_for(&source);
        let draft = standard_draft(vec![copy_operation(0, &source)]);
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
        assert_eq!(first.version, SCHEMA_VERSION);
        assert_eq!(
            first.plan_hash.to_string(),
            "9e235b1c2e5a9106d67fd79e6487552a9f95419d4bbe151fb9c84c72676f8649"
        );
        let encoded = serde_json::to_string(&first).unwrap();
        for field in ["keymat_binding", "norm_mode", "kv_cache_format"] {
            assert!(!encoded.contains(field));
        }
        first.verify_hash().unwrap();
    }

    #[test]
    fn missing_and_duplicate_outputs_are_rejected() {
        let source = descriptor("a");
        let mut duplicate = copy_operation(0, &source);
        duplicate.id = OperationId(1);
        let draft = standard_draft(vec![copy_operation(0, &source), duplicate]);
        let error = TransformPlan::from_draft(
            draft,
            ModelFingerprint::from_digest(blake3::hash(b"source")),
            vec![source.clone()],
            layout_for(&source),
            &TransformConfig::default(),
        );
        assert!(error.is_err());
    }

    #[test]
    fn unknown_input_is_rejected() {
        let source = descriptor("a");
        let mut operation = copy_operation(0, &source);
        operation.inputs[0].descriptor = descriptor("b");
        let error = TransformPlan::from_draft(
            standard_draft(vec![operation]),
            ModelFingerprint::from_digest(blake3::hash(b"source")),
            vec![source.clone()],
            layout_for(&source),
            &TransformConfig::default(),
        );
        assert!(error.is_err());
    }

    #[test]
    fn forward_dependencies_are_rejected() {
        let source = descriptor("a");
        let mut operation = copy_operation(0, &source);
        operation.dependencies = vec![OperationId(0)];
        let error = TransformPlan::from_draft(
            standard_draft(vec![operation]),
            ModelFingerprint::from_digest(blake3::hash(b"source")),
            vec![source.clone()],
            layout_for(&source),
            &TransformConfig::default(),
        );
        assert!(error.is_err());
    }
}
