use aloepri_core::{
    error::{CompilerError, Result},
    plan::{OutputLayout, OutputShard, OutputTensor},
    types::{ByteLength, ByteOffset, TensorDescriptor},
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub fn plan_output_layout(
    tensors: &[TensorDescriptor],
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
    let mut groups: Vec<Vec<TensorDescriptor>> = Vec::new();
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

#[cfg(test)]
mod tests {
    use super::*;
    use aloepri_core::types::{DType, ShardId, TensorLocation, TensorName, TensorShape};

    fn tensor(name: &str, length: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: TensorName::try_from(name).unwrap(),
            shape: TensorShape::new(vec![length]),
            dtype: DType::U8,
            byte_length: ByteLength(length),
            location: TensorLocation {
                shard: ShardId(0),
                offset: ByteOffset(0),
                length: ByteLength(length),
            },
        }
    }

    #[test]
    fn oversized_tensor_gets_own_shard() {
        let layout = plan_output_layout(&[tensor("a", 9), tensor("b", 1)], ByteLength(4)).unwrap();
        assert_eq!(layout.shards.len(), 2);
        assert_eq!(layout.shards[0].payload_length, ByteLength(9));
    }
}
