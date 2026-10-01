use aloepri_core::{
    ByteOffset, CompilerError, DType, ModelArtifact, Result, TensorReader, TensorSink,
    executor::{StreamingExecutor, TransformExecutor},
    keymat::KeyMatBinding,
    memory::MemoryBudget,
    plan::{KeyMatRole, MethodContract, Operation, OperationKind, TransformConfig},
};
use aloepri_secret::keymat::{KeyMatSecretV1, KeyMaterial};
use std::sync::Arc;

pub struct KeyMatExecutor {
    keys: Arc<KeyMaterial>,
    binding: KeyMatBinding,
}

impl KeyMatExecutor {
    pub fn new(secret: &KeyMatSecretV1, keys: Arc<KeyMaterial>) -> Result<Self> {
        secret.validate_material(&keys)?;
        Ok(Self {
            keys,
            binding: secret.binding()?,
        })
    }

    fn linear(
        &self,
        reader: &mut dyn TensorReader,
        op: &Operation,
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()> {
        let input = &op.inputs[0].descriptor;
        let output = &op.output.descriptor;
        if input.dtype != DType::F32
            || output.dtype != DType::F32
            || input.shape.as_slice().len() != 2
            || output.shape.as_slice().len() != 2
        {
            return Err(CompilerError::Unsupported(
                "KeyMat linear operations require F32 matrices".into(),
            ));
        }
        let d = self.keys.hidden_size();
        let big_d = self.keys.physical_hidden_size();
        let dimension = |value| {
            usize::try_from(value).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "KeyMat matrix dimension",
            })
        };
        let in_rows = dimension(input.shape.as_slice()[0])?;
        let in_cols = dimension(input.shape.as_slice()[1])?;
        let rows = dimension(output.shape.as_slice()[0])?;
        let cols = dimension(output.shape.as_slice()[1])?;
        let left = matches!(
            op.kind,
            OperationKind::KeyMatLeft {
                role: KeyMatRole::OutputPTranspose
            }
        );
        if (left && (in_rows != d || rows != big_d || cols != in_cols))
            || (!left && (in_cols != d || rows != in_rows || cols != big_d))
        {
            return Err(CompilerError::InvalidPlan {
                reason: "KeyMat executor geometry mismatch".into(),
            });
        }
        let width = (budget.available() / 16).min(256) as usize;
        if width == 0 {
            return Err(CompilerError::MemoryLimitExceeded {
                requested: 16,
                available: budget.available(),
            });
        }
        let _tile = budget.reserve((16 * width) as u64)?;
        let mut source = vec![0_u8; 4 * width];
        let mut acc = vec![0.0_f64; width];
        let mut encoded = vec![0_u8; 4 * width];
        for row in 0..rows {
            for col in (0..cols).step_by(width) {
                let count = width.min(cols - col);
                acc[..count].fill(0.0);
                if left {
                    for k in 0..d {
                        reader.read_bytes(
                            matrix_offset(k, col, in_cols)?,
                            &mut source[..count * 4],
                        )?;
                        let coefficient = self.keys.p()[k * big_d + row];
                        for (j, bytes) in source[..count * 4].as_chunks::<4>().0.iter().enumerate()
                        {
                            let value = f32::from_le_bytes(*bytes) as f64;
                            acc[j] += value * coefficient;
                        }
                    }
                } else {
                    for k_start in (0..d).step_by(width) {
                        let reduction = width.min(d - k_start);
                        reader.read_bytes(
                            matrix_offset(row, k_start, d)?,
                            &mut source[..reduction * 4],
                        )?;
                        for (offset, bytes) in source[..reduction * 4]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .enumerate()
                        {
                            let k = k_start + offset;
                            let value = f32::from_le_bytes(*bytes) as f64;
                            for (j, a) in acc[..count].iter_mut().enumerate() {
                                let coefficient = match op.kind {
                                    OperationKind::KeyMatRight {
                                        role: KeyMatRole::EmbeddingP,
                                    } => self.keys.p()[k * big_d + col + j],
                                    OperationKind::KeyMatRight {
                                        role:
                                            KeyMatRole::InputQTranspose | KeyMatRole::HeadQTranspose,
                                    } => self.keys.q()[(col + j) * d + k],
                                    _ => {
                                        return Err(CompilerError::InvalidPlan {
                                            reason: "invalid KeyMat linear role".into(),
                                        });
                                    }
                                };
                                *a += value * coefficient;
                            }
                        }
                    }
                }
                for j in 0..count {
                    let value = acc[j] as f32;
                    if !value.is_finite() {
                        return Err(CompilerError::InvalidTensor {
                            name: output.name.to_string(),
                            reason: "non-finite KeyMat result".into(),
                        });
                    }
                    encoded[j * 4..(j + 1) * 4].copy_from_slice(&value.to_le_bytes());
                }
                sink.write_bytes(&encoded[..count * 4])?;
            }
        }
        Ok(())
    }
}

fn matrix_offset(row: usize, column: usize, width: usize) -> Result<ByteOffset> {
    let overflow = || CompilerError::ArithmeticOverflow {
        operation: "KeyMat source byte offset",
    };
    let row = u64::try_from(row).map_err(|_| overflow())?;
    let column = u64::try_from(column).map_err(|_| overflow())?;
    let width = u64::try_from(width).map_err(|_| overflow())?;
    row.checked_mul(width)
        .and_then(|n| n.checked_add(column))
        .and_then(|n| n.checked_mul(4))
        .map(ByteOffset)
        .ok_or_else(overflow)
}

impl TransformExecutor for KeyMatExecutor {
    fn requirements(&self, config: &TransformConfig) -> Result<()> {
        if config.method != MethodContract::aloepri_keymat()
            || config.keymat_binding.as_ref() != Some(&self.binding)
            || config.secret_binding.is_some()
            || config.workers != 1
            || config.output_dtype != aloepri_core::OutputDType::Preserve
        {
            return Err(CompilerError::InvalidPlan {
                reason: "KeyMat executor/config binding mismatch".into(),
            });
        }
        Ok(())
    }
    fn execute_operation(
        &self,
        artifact: &dyn ModelArtifact,
        op: &Operation,
        sink: &mut dyn TensorSink,
        budget: &MemoryBudget,
    ) -> Result<()> {
        match op.kind {
            OperationKind::Copy => StreamingExecutor.execute_operation(artifact, op, sink, budget),
            OperationKind::KeyMatRight { .. } | OperationKind::KeyMatLeft { .. } => {
                let mut reader = artifact.tensor_reader(&op.inputs[0].descriptor.name)?;
                self.linear(reader.as_mut(), op, sink, budget)
            }
            _ => Err(CompilerError::Unsupported(
                "KeyMat executor does not implement this operation".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aloepri_core::{
        ByteLength, ModelFingerprint, OperationId, OperationInput, OperationOutput, ShardId,
        TensorDescriptor, TensorLocation, TensorName, TensorShape,
    };
    struct Reader {
        bytes: Vec<u8>,
        maximum: usize,
        limit: usize,
    }
    impl TensorReader for Reader {
        fn len(&self) -> ByteLength {
            ByteLength(self.bytes.len() as u64)
        }
        fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()> {
            assert!(destination.len() <= self.limit);
            self.maximum = self.maximum.max(destination.len());
            let start = offset.0 as usize;
            destination.copy_from_slice(&self.bytes[start..start + destination.len()]);
            Ok(())
        }
    }
    struct Sink(Vec<u8>);
    impl TensorSink for Sink {
        fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
            self.0.extend_from_slice(bytes);
            Ok(())
        }
        fn finish(self: Box<Self>) -> Result<ModelFingerprint> {
            Ok(ModelFingerprint::from_digest(blake3::hash(&self.0)))
        }
    }
    #[test]
    fn four_matrix_directions_stream_with_tiny_and_remainder_tiles() {
        assert_eq!(matrix_offset(2, 3, 5).unwrap(), ByteOffset(52));
        assert!(matrix_offset(usize::MAX, 0, usize::MAX).is_err());
        let (secret, keys, _) = KeyMatSecretV1::generate(
            ModelFingerprint::from_digest(blake3::hash(b"source")),
            3,
            2,
            0.3,
            Some([3; 32]),
        )
        .unwrap();
        let keys = Arc::new(keys);
        let executor = KeyMatExecutor::new(&secret, keys.clone()).unwrap();
        for role in [
            KeyMatRole::EmbeddingP,
            KeyMatRole::InputQTranspose,
            KeyMatRole::HeadQTranspose,
            KeyMatRole::OutputPTranspose,
        ] {
            let left = role == KeyMatRole::OutputPTranspose;
            let (in_rows, in_cols, out_rows, out_cols) =
                if left { (3, 5, 7, 5) } else { (5, 3, 5, 7) };
            let values: Vec<f32> = (0..in_rows * in_cols)
                .map(|i| (i as f32 - 7.0) / 17.0)
                .collect();
            let input = TensorDescriptor {
                name: TensorName::try_from("test").unwrap(),
                shape: TensorShape::new(vec![in_rows as u64, in_cols as u64]),
                dtype: DType::F32,
                byte_length: ByteLength((values.len() * 4) as u64),
                location: TensorLocation {
                    shard: ShardId(0),
                    offset: ByteOffset(0),
                    length: ByteLength((values.len() * 4) as u64),
                },
            };
            let mut output: aloepri_core::OutputTensorDescriptor = (&input).into();
            output.shape = TensorShape::new(vec![out_rows as u64, out_cols as u64]);
            output.byte_length = output.expected_byte_length().unwrap();
            let op = Operation {
                id: OperationId(0),
                kind: if left {
                    OperationKind::KeyMatLeft { role }
                } else {
                    OperationKind::KeyMatRight { role }
                },
                inputs: vec![OperationInput { descriptor: input }],
                output: OperationOutput { descriptor: output },
                memory_requirement: ByteLength(16),
                dependencies: vec![],
            };
            for bytes in [16, 48, 4096] {
                let mut reader = Reader {
                    bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    maximum: 0,
                    limit: (bytes / 16 * 4).min(1024) as usize,
                };
                let mut sink = Sink(vec![]);
                let budget = MemoryBudget::new(bytes);
                executor
                    .linear(&mut reader, &op, &mut sink, &budget)
                    .unwrap();
                assert_eq!(budget.used(), 0);
                let actual: Vec<f32> = sink
                    .0
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect();
                for i in 0..out_rows {
                    for j in 0..out_cols {
                        let expected: f64 = (0..3)
                            .map(|k| {
                                if left {
                                    keys.p()[k * 7 + i] * values[k * in_cols + j] as f64
                                } else if role == KeyMatRole::EmbeddingP {
                                    values[i * 3 + k] as f64 * keys.p()[k * 7 + j]
                                } else {
                                    values[i * 3 + k] as f64 * keys.q()[j * 3 + k]
                                }
                            })
                            .sum();
                        assert_eq!(actual[i * out_cols + j], expected as f32);
                    }
                }
                assert!(reader.maximum <= reader.limit);
            }
            let mut reader = Reader {
                bytes: vec![],
                maximum: 0,
                limit: 0,
            };
            let mut sink = Sink(vec![]);
            assert!(
                executor
                    .linear(&mut reader, &op, &mut sink, &MemoryBudget::new(15))
                    .is_err()
            );
        }
    }
}
