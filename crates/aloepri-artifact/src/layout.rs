use aloepri_core::{
    error::{CompilerError, Result},
    plan::{OutputLayout, OutputShard, OutputTensor},
    types::{ByteLength, ByteOffset, OutputTensorDescriptor},
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub fn plan_output_layout(
    tensors: &[OutputTensorDescriptor],
    max_shard_size: ByteLength,
) -> Result<OutputLayout> {
    if max_shard_size.0 == 0 {
        return Err(CompilerError::InvalidPlan {
            reason: "max shard size must be greater than zero".into(),
        });
    }
    let mut ordered = tensors.to_vec();
    ordered.sort_by(|left, right| {
        right
            .dtype
            .alignment()
            .cmp(&left.dtype.alignment())
            .then_with(|| left.name.cmp(&right.name))
    });
    let mut groups: Vec<Vec<OutputTensorDescriptor>> = Vec::new();
    let mut current = Vec::new();
    let mut current_size = 0_u64;
    for tensor in ordered {
        let length = tensor.byte_length.0;
        if !current.is_empty() && current_size.saturating_add(length) > max_shard_size.0 {
            groups.push(std::mem::take(&mut current));
            current_size = 0;
        }
        current_size =
            current_size
                .checked_add(length)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "output shard payload length",
                })?;
        current.push(tensor);
        if length > max_shard_size.0 {
            groups.push(std::mem::take(&mut current));
            current_size = 0;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    if groups.is_empty() {
        groups.push(Vec::new());
    }

    let multi = groups.len() > 1;
    let total = groups.len();
    let mut shards = Vec::new();
    let mut output_tensors = Vec::new();
    for (shard_index, group) in groups.iter().enumerate() {
        let filename = if multi {
            format!("model-{:05}-of-{:05}.safetensors", shard_index + 1, total)
        } else {
            "model.safetensors".into()
        };
        let mut entries = Map::new();
        let mut offset = 0_u64;
        for tensor in group {
            let end = offset.checked_add(tensor.byte_length.0).ok_or(
                CompilerError::ArithmeticOverflow {
                    operation: "output tensor range",
                },
            )?;
            entries.insert(
                tensor.name.to_string(),
                json!({
                    "dtype": tensor.dtype.as_safetensors(),
                    "shape": tensor.shape.as_slice(),
                    "data_offsets": [offset, end],
                }),
            );
            output_tensors.push(OutputTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                dtype: tensor.dtype,
                byte_length: tensor.byte_length,
                shard: shard_index as u32,
                offset: ByteOffset(offset),
            });
            offset = end;
        }
        let mut header = BTreeMap::<String, Value>::new();
        header.insert("__metadata__".into(), json!({"format": "pt"}));
        for (name, value) in entries {
            header.insert(name, value);
        }
        let json_header = serde_json::to_vec(&header)
            .map_err(|error| CompilerError::Invariant(error.to_string()))?;
        let padding = (8 - (json_header.len() % 8)) % 8;
        let mut header_bytes = json_header;
        header_bytes.resize(header_bytes.len() + padding, b' ');
        let header_length =
            u64::try_from(header_bytes.len()).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "output header length",
            })?;
        let file_length = 8_u64
            .checked_add(header_length)
            .and_then(|value| value.checked_add(offset))
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "output file length",
            })?;
        shards.push(OutputShard {
            id: shard_index as u32,
            filename,
            payload_length: ByteLength(offset),
            file_length: ByteLength(file_length),
            header: header_bytes,
        });
    }
    output_tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let index = if multi {
        let mut weight_map = BTreeMap::new();
        for tensor in &output_tensors {
            weight_map.insert(
                tensor.name.to_string(),
                shards[tensor.shard as usize].filename.clone(),
            );
        }
        let value = json!({
            "metadata": {"total_size": output_tensors.iter().map(|tensor| tensor.byte_length.0).sum::<u64>()},
            "weight_map": weight_map,
        });
        Some(
            serde_json::to_vec(&value)
                .map_err(|error| CompilerError::Invariant(error.to_string()))?,
        )
    } else {
        None
    };
    Ok(OutputLayout {
        shards,
        tensors: output_tensors,
        index,
    })
}

/// Check that a planned layout is internally consistent before any file exists.
///
/// This validates the generated header, relative offsets, payload and file
/// lengths, shard identity and the index against the layout tensors, so a
/// malformed plan fails closed rather than producing a corrupt artifact.
pub fn validate_output_layout(layout: &OutputLayout) -> Result<()> {
    if layout.shards.is_empty() {
        return Err(CompilerError::InvalidPlan {
            reason: "the output layout has no shards".into(),
        });
    }
    let mut tensors_by_shard: BTreeMap<u32, Vec<&OutputTensor>> = BTreeMap::new();
    for (index, shard) in layout.shards.iter().enumerate() {
        if shard.id as usize != index {
            return Err(CompilerError::InvalidPlan {
                reason: "shard ids must be dense and ordered".into(),
            });
        }
    }
    for tensor in &layout.tensors {
        let shard =
            layout
                .shards
                .get(tensor.shard as usize)
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: format!("output {} references an unknown shard", tensor.name),
                })?;
        if tensor.shard as usize >= layout.shards.len() || shard.id != tensor.shard {
            return Err(CompilerError::InvalidPlan {
                reason: format!("output {} is assigned to the wrong shard", tensor.name),
            });
        }
        let end = tensor.offset.0.checked_add(tensor.byte_length.0).ok_or(
            CompilerError::ArithmeticOverflow {
                operation: "output layout range",
            },
        )?;
        if end > shard.payload_length.0 {
            return Err(CompilerError::InvalidPlan {
                reason: format!("output {} exceeds its shard", tensor.name),
            });
        }
        tensors_by_shard
            .entry(tensor.shard)
            .or_default()
            .push(tensor);
    }
    for (index, shard) in layout.shards.iter().enumerate() {
        let expected_payload = tensors_by_shard
            .get(&(index as u32))
            .map(|tensors| {
                tensors.iter().try_fold(0_u64, |accumulator, tensor| {
                    accumulator.checked_add(tensor.byte_length.0).ok_or(
                        CompilerError::ArithmeticOverflow {
                            operation: "shard payload length",
                        },
                    )
                })
            })
            .transpose()?
            .unwrap_or(0);
        if expected_payload != shard.payload_length.0 {
            return Err(CompilerError::InvalidPlan {
                reason: format!("shard {} payload length is inconsistent", shard.filename),
            });
        }
        let header_length =
            u64::try_from(shard.header.len()).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "shard header length",
            })?;
        let expected_file_length = 8_u64
            .checked_add(header_length)
            .and_then(|value| value.checked_add(shard.payload_length.0))
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "shard file length",
            })?;
        if expected_file_length != shard.file_length.0 {
            return Err(CompilerError::InvalidPlan {
                reason: format!("shard {} file length is inconsistent", shard.filename),
            });
        }
        validate_header(shard, index as u32, &tensors_by_shard)?;
    }
    if let Some(index) = &layout.index {
        let value: Value =
            serde_json::from_slice(index).map_err(|error| CompilerError::InvalidPlan {
                reason: format!("output index is not valid JSON: {error}"),
            })?;
        let weight_map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| CompilerError::InvalidPlan {
                reason: "output index has no weight_map".into(),
            })?;
        if weight_map.len() != layout.tensors.len() {
            return Err(CompilerError::InvalidPlan {
                reason: "output index does not list every output".into(),
            });
        }
        let total_size = value
            .get("metadata")
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(Value::as_u64);
        let expected_total: u64 =
            layout
                .tensors
                .iter()
                .try_fold(0_u64, |accumulator, tensor| {
                    accumulator.checked_add(tensor.byte_length.0).ok_or(
                        CompilerError::ArithmeticOverflow {
                            operation: "index total size",
                        },
                    )
                })?;
        if total_size != Some(expected_total) {
            return Err(CompilerError::InvalidPlan {
                reason: "output index total_size is inconsistent".into(),
            });
        }
        for tensor in &layout.tensors {
            let entry = weight_map
                .get(tensor.name.as_str())
                .and_then(Value::as_str)
                .ok_or_else(|| CompilerError::InvalidPlan {
                    reason: format!("output index is missing {}", tensor.name),
                })?;
            if entry != layout.shards[tensor.shard as usize].filename {
                return Err(CompilerError::InvalidPlan {
                    reason: format!("output index maps {} to the wrong shard", tensor.name),
                });
            }
        }
    }
    Ok(())
}

fn validate_header(
    shard: &OutputShard,
    shard_id: u32,
    tensors_by_shard: &BTreeMap<u32, Vec<&OutputTensor>>,
) -> Result<()> {
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&shard.header).map_err(|error| CompilerError::InvalidPlan {
            reason: format!("shard {} header is not valid JSON: {error}", shard.filename),
        })?;
    let mut expected: BTreeMap<&str, &OutputTensor> = BTreeMap::new();
    for tensor in tensors_by_shard.get(&shard_id).into_iter().flatten() {
        expected.insert(tensor.name.as_str(), tensor);
    }
    if header.len() != expected.len() + 1 {
        return Err(CompilerError::InvalidPlan {
            reason: format!(
                "shard {} header entry count is inconsistent",
                shard.filename
            ),
        });
    }
    for (name, value) in &header {
        if name == "__metadata__" {
            continue;
        }
        let tensor = expected
            .get(name.as_str())
            .ok_or_else(|| CompilerError::InvalidPlan {
                reason: format!(
                    "shard {} header has an unplanned tensor {name}",
                    shard.filename
                ),
            })?;
        let dtype = value.get("dtype").and_then(Value::as_str);
        if dtype != Some(tensor.dtype.as_safetensors()) {
            return Err(CompilerError::InvalidPlan {
                reason: format!("shard {} header dtype differs for {name}", shard.filename),
            });
        }
        let shape: Vec<u64> = value
            .get("shape")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        if shape != tensor.shape.as_slice() {
            return Err(CompilerError::InvalidPlan {
                reason: format!("shard {} header shape differs for {name}", shard.filename),
            });
        }
        let offsets = value.get("data_offsets").and_then(Value::as_array);
        let start = offsets
            .and_then(|values| values.first())
            .and_then(Value::as_u64);
        let end = offsets
            .and_then(|values| values.get(1))
            .and_then(Value::as_u64);
        if start != Some(tensor.offset.0) || end != Some(tensor.offset.0 + tensor.byte_length.0) {
            return Err(CompilerError::InvalidPlan {
                reason: format!("shard {} header offsets differ for {name}", shard.filename),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aloepri_core::types::{DType, TensorName, TensorShape};

    fn tensor(name: &str, length: u64) -> OutputTensorDescriptor {
        OutputTensorDescriptor {
            name: TensorName::try_from(name).unwrap(),
            shape: TensorShape::new(vec![length]),
            dtype: DType::U8,
            byte_length: ByteLength(length),
        }
    }

    #[test]
    fn oversized_tensor_gets_own_shard() {
        let layout = plan_output_layout(&[tensor("a", 9), tensor("b", 1)], ByteLength(4)).unwrap();
        assert_eq!(layout.shards.len(), 2);
        assert_eq!(layout.shards[0].payload_length, ByteLength(9));
        validate_output_layout(&layout).unwrap();
    }

    #[test]
    fn shape_change_re_shards_outputs() {
        let small = plan_output_layout(&[tensor("a", 2)], ByteLength(4)).unwrap();
        assert_eq!(small.shards.len(), 1);
        let grown = plan_output_layout(&[tensor("a", 6), tensor("b", 6)], ByteLength(6)).unwrap();
        assert_eq!(grown.shards.len(), 2);
        assert!(grown.index.is_some());
        validate_output_layout(&grown).unwrap();
    }

    #[test]
    fn tampered_layout_header_is_rejected() {
        let mut layout = plan_output_layout(&[tensor("a", 2)], ByteLength(4)).unwrap();
        layout.shards[0].header = b"{}".to_vec();
        assert!(validate_output_layout(&layout).is_err());
    }
}
