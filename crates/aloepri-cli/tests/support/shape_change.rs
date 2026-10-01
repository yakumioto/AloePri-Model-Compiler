//! Diagnostic shape-changing compiler harness used by the integration tests.
//!
//! This is test-only scaffolding: the product CLI never exposes `expand-test`,
//! and the padding executor exists solely to prove the pipeline no longer
//! assumes `input shape == output shape`. It is not the AloePri embedding
//! algorithm and makes no claim to be one.

#![allow(dead_code)]

use aloepri_core::{
    Compiler,
    error::{CompilerError, Result},
    executor::{StreamingExecutor, TransformExecutor},
    io::{TensorReader, TensorSink, copy_bytes},
    memory::MemoryBudget,
    model::{ArchitectureAdapter, ArchitectureRegistry, ModelArtifact},
    plan::{
        MethodContract, Operation, OperationInput, OperationKind, OperationOutput, PlanDraft,
        RuntimeContract, TransformConfig,
    },
    types::{ByteLength, ByteOffset, OutputDType},
};
use std::collections::BTreeMap;

/// F32 padding of a single row: `source_row || zeros`.
///
/// The reader is bounded: a request is split at the row boundary so a read
/// never materializes a whole row or the whole tensor.
struct PaddedReader {
    source: Box<dyn TensorReader>,
    source_row_bytes: u64,
    output_row_bytes: u64,
    length: ByteLength,
}

impl PaddedReader {
    fn new(
        source: Box<dyn TensorReader>,
        source_row_bytes: u64,
        output_row_bytes: u64,
        length: ByteLength,
    ) -> Result<Self> {
        if source_row_bytes == 0
            || output_row_bytes < source_row_bytes
            || !length.0.is_multiple_of(output_row_bytes)
        {
            return Err(CompilerError::InvalidTensor {
                name: "pad columns".into(),
                reason: "invalid padded reader geometry".into(),
            });
        }
        Ok(Self {
            source,
            source_row_bytes,
            output_row_bytes,
            length,
        })
    }
}

impl TensorReader for PaddedReader {
    fn len(&self) -> ByteLength {
        self.length
    }

    fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()> {
        let requested =
            u64::try_from(destination.len()).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "padded reader length",
            })?;
        let end = offset
            .0
            .checked_add(requested)
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "padded reader range",
            })?;
        if end > self.length.0 {
            return Err(CompilerError::InvalidTensor {
                name: "pad columns".into(),
                reason: "reader range exceeds the output length".into(),
            });
        }
        let mut position = 0_usize;
        while position < destination.len() {
            let absolute = offset.0 + position as u64;
            let row = absolute / self.output_row_bytes;
            let within = absolute % self.output_row_bytes;
            let remaining = (destination.len() - position) as u64;
            if within < self.source_row_bytes {
                let available = self.source_row_bytes - within;
                let count = available.min(remaining) as usize;
                let source_offset = row
                    .checked_mul(self.source_row_bytes)
                    .and_then(|value| value.checked_add(within))
                    .ok_or(CompilerError::ArithmeticOverflow {
                        operation: "padded reader source offset",
                    })?;
                self.source.read_bytes(
                    ByteOffset(source_offset),
                    &mut destination[position..position + count],
                )?;
                position += count;
            } else {
                let available = self.output_row_bytes - within;
                let count = available.min(remaining) as usize;
                destination[position..position + count].fill(0);
                position += count;
            }
        }
        Ok(())
    }
}

/// A deliberate mutation of the diagnostic runtime contract, used to prove a
/// changed runtime cannot reuse an interrupted checkpoint.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum Tamper {
    #[default]
    None,
    /// A different runtime version: still structurally valid, different hash.
    RuntimeVersion,
    /// An extra physical dimension: still structurally valid, different hash.
    ExtraPhysicalDimension,
    /// Claim the artifact is a standard HF checkpoint: rejected outright.
    StandardFlag,
}

/// Registry that recognises the diagnostic `expand-test` architecture.
pub struct ShapeTestRegistry {
    tamper: Tamper,
}

impl ShapeTestRegistry {
    pub fn new(tamper: Tamper) -> Self {
        Self { tamper }
    }
}

impl Default for ShapeTestRegistry {
    fn default() -> Self {
        Self {
            tamper: Tamper::None,
        }
    }
}

impl ArchitectureRegistry for ShapeTestRegistry {
    fn detect(&self, artifact: &dyn ModelArtifact) -> Result<Box<dyn ArchitectureAdapter>> {
        if artifact.model_spec().architecture == "expand-test" {
            return Ok(Box::new(ShapeTestAdapter {
                tamper: self.tamper,
            }));
        }
        Err(CompilerError::UnsupportedArchitecture {
            reason: "the diagnostic adapter only handles expand-test models".into(),
        })
    }
}

/// Adapter that pads the last dimension of every two-dimensional tensor.
pub struct ShapeTestAdapter {
    tamper: Tamper,
}

/// Decouple an output name from the source name it is read from: any tensor
/// ending in `.source.weight` is published as `.target.weight`.
fn output_name(input: &aloepri_core::types::TensorName) -> aloepri_core::types::TensorName {
    match input.as_str().strip_suffix(".source.weight") {
        Some(prefix) => aloepri_core::types::TensorName::new(format!("{prefix}.target.weight"))
            .expect("fixture tensor names are valid"),
        None => input.clone(),
    }
}

impl ShapeTestAdapter {
    fn expansion(artifact: &dyn ModelArtifact) -> Result<u64> {
        artifact
            .config()
            .get("expansion_size")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| CompilerError::UnsupportedArchitecture {
                reason: "expand-test config needs a positive expansion_size".into(),
            })
    }

    fn logical_hidden(artifact: &dyn ModelArtifact) -> Result<u64> {
        artifact
            .config()
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| CompilerError::UnsupportedArchitecture {
                reason: "expand-test config needs hidden_size".into(),
            })
    }
}

impl ArchitectureAdapter for ShapeTestAdapter {
    fn id(&self) -> &'static str {
        "expand-test"
    }

    fn build_plan(
        &self,
        artifact: &dyn ModelArtifact,
        _config: &TransformConfig,
    ) -> Result<PlanDraft> {
        let expansion = Self::expansion(artifact)?;
        let logical_hidden = Self::logical_hidden(artifact)?;
        let additional_columns = expansion * 2;
        let mut operations = Vec::new();
        for (index, tensor) in artifact.tensors().iter().enumerate() {
            let columns = tensor.shape.as_slice().get(1).copied();
            let kind = if tensor.shape.as_slice().len() == 2 && columns == Some(logical_hidden) {
                OperationKind::PadColumns { additional_columns }
            } else {
                OperationKind::Copy
            };
            let output = match kind {
                OperationKind::PadColumns { additional_columns } => {
                    let mut shape = tensor.shape.as_slice().to_vec();
                    shape[1] += additional_columns;
                    let byte_length = tensor
                        .shape
                        .element_count()?
                        .checked_add(
                            tensor.shape.as_slice()[0]
                                .checked_mul(additional_columns)
                                .ok_or(CompilerError::ArithmeticOverflow {
                                    operation: "padded element count",
                                })?,
                        )
                        .ok_or(CompilerError::ArithmeticOverflow {
                            operation: "padded element count",
                        })?
                        .checked_mul(tensor.dtype.byte_width())
                        .ok_or(CompilerError::ArithmeticOverflow {
                            operation: "padded byte length",
                        })?;
                    aloepri_core::types::OutputTensorDescriptor {
                        name: output_name(&tensor.name),
                        shape: aloepri_core::types::TensorShape::new(shape),
                        dtype: tensor.dtype,
                        byte_length: ByteLength(byte_length),
                    }
                }
                OperationKind::Copy => tensor.into(),
                OperationKind::TokenPermutation { .. }
                | OperationKind::KeyMatRight { .. }
                | OperationKind::KeyMatLeft { .. } => {
                    unreachable!("the diagnostic adapter never plans a token permutation")
                }
            };
            operations.push(Operation {
                id: aloepri_core::types::OperationId(index as u32),
                kind,
                inputs: vec![OperationInput {
                    descriptor: tensor.clone(),
                }],
                output: OperationOutput { descriptor: output },
                memory_requirement: ByteLength(4096),
                dependencies: Vec::new(),
            });
        }
        let logical_dimensions = logical_dimensions(artifact);
        let mut physical_dimensions = logical_dimensions.clone();
        let physical_hidden = logical_hidden + 2 * expansion;
        physical_dimensions.insert("hidden_size".into(), physical_hidden);
        let mut runtime_contract = RuntimeContract {
            id: "expand-test-runtime".into(),
            version: "0.1".into(),
            standard_hf_checkpoint: false,
            architecture: "expand-test".into(),
            logical_dimensions,
            physical_dimensions,
            expansion_size: Some(expansion),
            norm_mode: None,
            kv_cache_format: None,
        };
        match self.tamper {
            Tamper::None => {}
            Tamper::RuntimeVersion => runtime_contract.version = "9.9".into(),
            Tamper::ExtraPhysicalDimension => {
                runtime_contract
                    .physical_dimensions
                    .insert("smuggled_dimension".into(), 7);
            }
            Tamper::StandardFlag => runtime_contract.standard_hf_checkpoint = true,
        }
        Ok(PlanDraft {
            architecture: "expand-test".into(),
            runtime_contract,
            operations,
        })
    }
}

fn logical_dimensions(artifact: &dyn ModelArtifact) -> BTreeMap<String, u64> {
    let config = artifact.config();
    let mut dimensions = BTreeMap::new();
    for key in ["hidden_size", "vocab_size"] {
        if let Some(value) = config.get(key).and_then(serde_json::Value::as_u64) {
            dimensions.insert(key.to_owned(), value);
        }
    }
    dimensions
}

/// Executor that realises `PadColumns` and delegates copies.
#[derive(Default)]
pub struct PadColumnsExecutor {
    copy: StreamingExecutor,
}

impl TransformExecutor for PadColumnsExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()> {
        if config.method != MethodContract::expand_test() {
            return Err(CompilerError::Unsupported(
                "the padding executor requires expand-test/0.1".into(),
            ));
        }
        if config.workers != 1 {
            return Err(CompilerError::Unsupported(
                "expand-test uses a single worker".into(),
            ));
        }
        if config.output_dtype != OutputDType::Preserve {
            return Err(CompilerError::Unsupported(
                "expand-test preserves the source dtype".into(),
            ));
        }
        Ok(())
    }

    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()> {
        match operation.kind {
            OperationKind::Copy => self
                .copy
                .execute_operation(artifact, operation, sink, budget),
            OperationKind::PadColumns { additional_columns } => {
                let input = &operation.inputs[0].descriptor;
                let output = &operation.output.descriptor;
                let width = input.dtype.byte_width();
                let columns = input.shape.as_slice().get(1).copied().ok_or_else(|| {
                    CompilerError::InvalidTensor {
                        name: input.name.to_string(),
                        reason: "padding needs a two-dimensional tensor".into(),
                    }
                })?;
                let source_row_bytes = columns * width;
                let output_row_bytes = source_row_bytes + additional_columns * width;
                let source = artifact.tensor_reader(&input.name)?;
                let mut reader = PaddedReader::new(
                    source,
                    source_row_bytes,
                    output_row_bytes,
                    output.byte_length,
                )?;
                let mut write = |bytes: &[u8]| sink.write_bytes(bytes);
                copy_bytes(&mut reader, &mut write, output.byte_length, budget)?;
                Ok(())
            }
            OperationKind::TokenPermutation { .. }
            | OperationKind::KeyMatRight { .. }
            | OperationKind::KeyMatLeft { .. } => Err(CompilerError::Unsupported(
                "the diagnostic executor does not implement token permutation".into(),
            )),
        }
    }
}

/// Fires a deterministic failure during the `fail_at`-th executed operation.
pub struct FaultInjectingExecutor {
    inner: PadColumnsExecutor,
    fail_at: usize,
    seen: std::cell::Cell<usize>,
}

impl FaultInjectingExecutor {
    pub fn new(fail_at: usize) -> Self {
        Self {
            inner: PadColumnsExecutor::default(),
            fail_at,
            seen: std::cell::Cell::new(0),
        }
    }
}

impl TransformExecutor for FaultInjectingExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()> {
        self.inner.requirements(config)
    }

    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()> {
        let index = self.seen.get();
        self.seen.set(index + 1);
        if index == self.fail_at {
            return Err(CompilerError::Invariant("injected executor failure".into()));
        }
        self.inner
            .execute_operation(artifact, operation, sink, budget)
    }
}

pub type ShapeCompiler<S> = Compiler<aloepri_artifact::HfBackend, ShapeTestRegistry, S>;

pub fn shape_compiler<S: TransformExecutor>(executor: S) -> ShapeCompiler<S> {
    shape_compiler_with(Tamper::None, executor)
}

pub fn shape_compiler_with<S: TransformExecutor>(tamper: Tamper, executor: S) -> ShapeCompiler<S> {
    Compiler::new(
        aloepri_artifact::HfBackend,
        ShapeTestRegistry::new(tamper),
        executor,
    )
}
