use aloepri_core::{
    error::{CompilerError, Result},
    executor::{StreamingExecutor, TransformExecutor},
    io::{TensorReader, TensorSink, copy_bytes},
    memory::MemoryBudget,
    model::ModelArtifact,
    plan::{MethodContract, Operation, OperationKind, SecretBinding, TokenRole, TransformConfig},
    types::{ByteLength, ByteOffset, DType, ModelFingerprint, TensorShape},
};

pub struct TokenPermutationExecutor {
    inverse_permutation: Vec<u32>,
    binding: SecretBinding,
    copy: StreamingExecutor,
}

impl TokenPermutationExecutor {
    pub fn new(inverse_permutation: Vec<u32>, binding: SecretBinding) -> Result<Self> {
        let length =
            usize::try_from(binding.vocab_size).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "token permutation vocabulary conversion",
            })?;
        if length < 2 || inverse_permutation.len() != length {
            return Err(CompilerError::InvalidPlan {
                reason: "token inverse permutation length does not match vocabulary".into(),
            });
        }
        let mut seen = vec![false; length];
        for &value in &inverse_permutation {
            let index = usize::try_from(value).map_err(|_| CompilerError::InvalidPlan {
                reason: "token inverse permutation contains an invalid token id".into(),
            })?;
            if index >= length || seen[index] {
                return Err(CompilerError::InvalidPlan {
                    reason: "token inverse permutation is not a bijection".into(),
                });
            }
            seen[index] = true;
        }
        Ok(Self {
            inverse_permutation,
            binding,
            copy: StreamingExecutor,
        })
    }

    pub fn binding(&self) -> &SecretBinding {
        &self.binding
    }

    pub fn inverse_permutation(&self) -> &[u32] {
        &self.inverse_permutation
    }
}

impl TransformExecutor for TokenPermutationExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()> {
        if config.method != MethodContract::aloepri_token() {
            return Err(CompilerError::Unsupported(
                "token executor requires aloepri-token/0.1".into(),
            ));
        }
        if config.workers != 1 {
            return Err(CompilerError::Unsupported(
                "aloepri-token uses a single worker".into(),
            ));
        }
        if config.output_dtype != aloepri_core::types::OutputDType::Preserve {
            return Err(CompilerError::Unsupported(
                "aloepri-token preserves the source dtype".into(),
            ));
        }
        if config.secret_binding.as_ref() != Some(&self.binding) {
            return Err(CompilerError::InvalidPlan {
                reason: "token executor binding does not match the transform plan".into(),
            });
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
            OperationKind::TokenPermutation { role } => {
                self.execute_permutation(artifact, operation, role, sink, budget)
            }
            OperationKind::PadColumns { .. } => Err(CompilerError::Unsupported(
                "aloepri-token does not implement column padding".into(),
            )),
        }
    }
}

impl TokenPermutationExecutor {
    fn execute_permutation(
        &self,
        artifact: &dyn ModelArtifact,
        operation: &Operation,
        role: TokenRole,
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()> {
        let source_descriptor = &operation.inputs[0].descriptor;
        let name = &source_descriptor.name;
        let row_bytes = row_bytes(&source_descriptor.shape, source_descriptor.dtype)?;
        let vocab_size = source_descriptor
            .shape
            .as_slice()
            .first()
            .copied()
            .ok_or_else(|| CompilerError::InvalidTensor {
                name: name.to_string(),
                reason: "token tensor must have a vocabulary dimension".into(),
            })?;
        if vocab_size != self.binding.vocab_size {
            return Err(CompilerError::InvalidTensor {
                name: name.to_string(),
                reason: "token tensor vocabulary does not match the client secret".into(),
            });
        }
        let expected_length =
            row_bytes
                .checked_mul(vocab_size)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "token tensor byte length",
                })?;
        if expected_length != source_descriptor.byte_length.0 {
            return Err(CompilerError::InvalidTensor {
                name: name.to_string(),
                reason: "token tensor byte length does not match its shape".into(),
            });
        }
        let mapping_bytes =
            self.binding
                .vocab_size
                .checked_mul(8)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "token permutation memory reservation",
                })?;
        let mapping_reservation = budget.reserve(mapping_bytes)?;
        let mut source_for_hash = artifact.tensor_reader(name)?;
        let source_digest =
            fingerprint_reader(source_for_hash.as_mut(), source_descriptor.byte_length)?;
        let source = artifact.tensor_reader(name)?;
        let mut reader = PermutedTensorReader::new(
            source,
            self.inverse_permutation.clone(),
            row_bytes,
            source_descriptor.byte_length,
        )?;
        // The executor hashes what it emits so it can prove the embedding
        // really changed; the authoritative completion digest still comes from
        // the sink, which the compiler finishes.
        let mut emitted = blake3::Hasher::new();
        let mut write = |bytes: &[u8]| {
            emitted.update(bytes);
            sink.write_bytes(bytes)
        };
        copy_bytes(
            &mut reader,
            &mut write,
            operation.output.descriptor.byte_length,
            budget,
        )?;
        let output_digest = ModelFingerprint::from_digest(emitted.finalize());
        drop(mapping_reservation);
        if matches!(role, TokenRole::InputEmbedding) && output_digest == source_digest {
            return Err(CompilerError::InvalidPlan {
                reason: "token permutation did not change embedding bytes".into(),
            });
        }
        Ok(())
    }
}

struct PermutedTensorReader {
    source: Box<dyn TensorReader>,
    inverse_permutation: Vec<u32>,
    row_bytes: u64,
    length: ByteLength,
}

impl PermutedTensorReader {
    fn new(
        source: Box<dyn TensorReader>,
        inverse_permutation: Vec<u32>,
        row_bytes: u64,
        length: ByteLength,
    ) -> Result<Self> {
        if row_bytes == 0 || source.len() != length {
            return Err(CompilerError::InvalidTensor {
                name: "token permutation".into(),
                reason: "invalid source reader or row width".into(),
            });
        }
        if !length.0.is_multiple_of(row_bytes)
            || inverse_permutation.len() as u64 != length.0 / row_bytes
        {
            return Err(CompilerError::InvalidTensor {
                name: "token permutation".into(),
                reason: "row width does not cover the source tensor".into(),
            });
        }
        Ok(Self {
            source,
            inverse_permutation,
            row_bytes,
            length,
        })
    }
}

impl TensorReader for PermutedTensorReader {
    fn len(&self) -> ByteLength {
        self.length
    }

    fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()> {
        let requested =
            u64::try_from(destination.len()).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "token reader destination length",
            })?;
        let end = offset
            .0
            .checked_add(requested)
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "token reader range",
            })?;
        if end > self.length.0 {
            return Err(CompilerError::InvalidTensor {
                name: "token permutation".into(),
                reason: "reader range exceeds tensor length".into(),
            });
        }
        let mut position = 0_usize;
        while position < destination.len() {
            let absolute = offset.0 + position as u64;
            let destination_row = absolute / self.row_bytes;
            let row_offset = absolute % self.row_bytes;
            let available = self.row_bytes - row_offset;
            let count = available.min((destination.len() - position) as u64) as usize;
            let source_row = *self
                .inverse_permutation
                .get(destination_row as usize)
                .ok_or_else(|| CompilerError::InvalidTensor {
                    name: "token permutation".into(),
                    reason: "reader selected an invalid destination row".into(),
                })? as u64;
            let source_offset = source_row
                .checked_mul(self.row_bytes)
                .and_then(|value| value.checked_add(row_offset))
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "token reader source offset",
                })?;
            self.source.read_bytes(
                ByteOffset(source_offset),
                &mut destination[position..position + count],
            )?;
            position += count;
        }
        Ok(())
    }
}

fn row_bytes(shape: &TensorShape, dtype: DType) -> Result<u64> {
    let hidden = shape
        .as_slice()
        .get(1)
        .copied()
        .ok_or_else(|| CompilerError::InvalidTensor {
            name: "token permutation".into(),
            reason: "token tensor must be two-dimensional".into(),
        })?;
    if hidden == 0 {
        return Err(CompilerError::InvalidTensor {
            name: "token permutation".into(),
            reason: "token tensor hidden dimension must be nonzero".into(),
        });
    }
    hidden
        .checked_mul(dtype.byte_width())
        .ok_or(CompilerError::ArithmeticOverflow {
            operation: "token tensor row width",
        })
}

fn fingerprint_reader(
    reader: &mut dyn TensorReader,
    length: ByteLength,
) -> Result<ModelFingerprint> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut offset = 0_u64;
    while offset < length.0 {
        let size = (length.0 - offset).min(buffer.len() as u64) as usize;
        reader.read_bytes(ByteOffset(offset), &mut buffer[..size])?;
        hasher.update(&buffer[..size]);
        offset += size as u64;
    }
    Ok(ModelFingerprint::from_digest(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aloepri_core::types::{ShardId, TensorDescriptor, TensorLocation};
    use std::io::Cursor;

    struct Reader(Cursor<Vec<u8>>);

    impl TensorReader for Reader {
        fn len(&self) -> ByteLength {
            ByteLength(self.0.get_ref().len() as u64)
        }

        fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()> {
            use std::io::Read;
            self.0.set_position(offset.0);
            self.0
                .read_exact(destination)
                .map_err(|error| CompilerError::Invariant(error.to_string()))
        }
    }

    #[test]
    fn reader_applies_a_non_self_inverse_cycle() {
        let mut reader = PermutedTensorReader::new(
            Box::new(Reader(Cursor::new(vec![10, 11, 20, 21, 30, 31]))),
            vec![2, 0, 1],
            2,
            ByteLength(6),
        )
        .unwrap();
        let mut output = [0_u8; 6];
        reader.read_bytes(ByteOffset(0), &mut output).unwrap();
        assert_eq!(output, [30, 31, 10, 11, 20, 21]);
        let mut partial = [0_u8; 3];
        reader.read_bytes(ByteOffset(1), &mut partial).unwrap();
        assert_eq!(partial, [31, 10, 11]);
    }

    #[test]
    fn row_width_uses_dtype_bytes() {
        assert_eq!(
            row_bytes(&TensorShape::new(vec![2, 3]), DType::BF16).unwrap(),
            6
        );
    }

    #[allow(dead_code)]
    fn descriptor() -> TensorDescriptor {
        TensorDescriptor {
            name: aloepri_core::types::TensorName::try_from("x").unwrap(),
            shape: TensorShape::new(vec![3, 2]),
            dtype: DType::F32,
            byte_length: ByteLength(24),
            location: TensorLocation {
                shard: ShardId(0),
                offset: ByteOffset(0),
                length: ByteLength(24),
            },
        }
    }
}
