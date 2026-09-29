use aloepri_core::{
    error::{CompilerError, Result, io_error, json_error},
    types::{ByteLength, DType, TensorName, TensorShape},
};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

pub const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct HeaderTensor {
    pub name: TensorName,
    pub shape: TensorShape,
    pub dtype: DType,
    pub relative_offset: u64,
    pub byte_length: ByteLength,
}

#[derive(Clone, Debug)]
pub struct ShardHeader {
    pub path: PathBuf,
    pub header_length: u64,
    pub data_start: u64,
    pub file_length: u64,
    pub tensors: Vec<HeaderTensor>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub fn read_safetensors_header(path: &Path) -> Result<ShardHeader> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let file_length = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();
    if file_length < 8 {
        return Err(CompilerError::InvalidArtifact {
            path: path.to_owned(),
            reason: "file is shorter than the safetensors header length prefix".into(),
        });
    }
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|source| io_error(path, source))?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length > MAX_HEADER_BYTES {
        return Err(CompilerError::InvalidArtifact {
            path: path.to_owned(),
            reason: format!("header exceeds {MAX_HEADER_BYTES} bytes"),
        });
    }
    let data_start = 8_u64
        .checked_add(header_length)
        .ok_or(CompilerError::ArithmeticOverflow {
            operation: "safetensors data start",
        })?;
    if data_start > file_length {
        return Err(CompilerError::InvalidArtifact {
            path: path.to_owned(),
            reason: "header extends beyond file".into(),
        });
    }
    let header_size =
        usize::try_from(header_length).map_err(|_| CompilerError::ArithmeticOverflow {
            operation: "header length conversion",
        })?;
    let mut bytes = vec![0_u8; header_size];
    file.read_exact(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    let json_bytes = bytes.trim_ascii_end();
    reject_duplicate_keys(json_bytes).map_err(|error| CompilerError::InvalidArtifact {
        path: path.to_owned(),
        reason: format!("duplicate or invalid header JSON: {error}"),
    })?;
    let value: BTreeMap<String, Value> =
        serde_json::from_slice(json_bytes).map_err(|source| json_error(path, source))?;
    let data_length = file_length - data_start;
    let mut tensors = Vec::new();
    let mut names = BTreeSet::new();
    let mut metadata = BTreeMap::new();
    for (name, value) in value {
        if name == "__metadata__" {
            let object = value
                .as_object()
                .ok_or_else(|| CompilerError::InvalidArtifact {
                    path: path.to_owned(),
                    reason: "__metadata__ must be an object".into(),
                })?;
            metadata.extend(
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
            continue;
        }
        if !names.insert(name.clone()) {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("duplicate tensor name {name}"),
            });
        }
        let raw: RawTensor =
            serde_json::from_value(value).map_err(|source| CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("invalid tensor {name}: {source}"),
            })?;
        let [start, end] = raw.data_offsets;
        if end < start || end > data_length {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("invalid data range for {name}: [{start}, {end}]"),
            });
        }
        let dtype = DType::from_safetensors(&raw.dtype)?;
        let shape = TensorShape::new(raw.shape);
        let expected = shape
            .element_count()?
            .checked_mul(dtype.byte_width())
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "safetensors tensor byte length",
            })?;
        if expected != end - start {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!(
                    "shape and dtype imply {expected} bytes for {name}, range contains {}",
                    end - start
                ),
            });
        }
        tensors.push(HeaderTensor {
            name: TensorName::try_from(name)?,
            shape,
            dtype,
            relative_offset: start,
            byte_length: ByteLength(end - start),
        });
    }
    tensors.sort_by(|left, right| {
        left.relative_offset
            .cmp(&right.relative_offset)
            .then_with(|| left.byte_length.0.cmp(&right.byte_length.0))
            .then_with(|| left.name.cmp(&right.name))
    });
    let mut cursor = 0_u64;
    for tensor in &tensors {
        if tensor.relative_offset != cursor {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!(
                    "tensor ranges contain a hole or overlap near {}",
                    tensor.name
                ),
            });
        }
        cursor =
            cursor
                .checked_add(tensor.byte_length.0)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "safetensors tensor ranges",
                })?;
    }
    if cursor != data_length {
        return Err(CompilerError::InvalidArtifact {
            path: path.to_owned(),
            reason: format!("tensor ranges cover {cursor} bytes, file has {data_length}"),
        });
    }
    file.seek(SeekFrom::Start(data_start))
        .map_err(|source| io_error(path, source))?;
    Ok(ShardHeader {
        path: path.to_owned(),
        header_length,
        data_start,
        file_length,
        tensors,
        metadata,
    })
}

pub fn reject_duplicate_keys(bytes: &[u8]) -> std::result::Result<(), String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    deserializer
        .deserialize_any(AnyVisitor)
        .map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())
}

struct AnySeed;
struct AnyVisitor;

impl<'de> DeserializeSeed<'de> for AnySeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(AnyVisitor)
    }
}

impl<'de> Visitor<'de> for AnyVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_map<M>(self, mut map: M) -> std::result::Result<(), M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!("duplicate key {key}")));
            }
            map.next_value_seed(AnySeed)?;
        }
        Ok(())
    }

    fn visit_seq<S>(self, mut sequence: S) -> std::result::Result<(), S::Error>
    where
        S: SeqAccess<'de>,
    {
        while sequence.next_element_seed(AnySeed)?.is_some() {}
        Ok(())
    }

    fn visit_bool<E>(self, _: bool) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _: f64) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_str<E>(self, _: &str) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_string<E>(self, _: String) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_none<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> std::result::Result<(), E> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn rejects_duplicate_json_keys() {
        assert!(reject_duplicate_keys(br#"{"a": 1, "a": 2}"#).is_err());
    }

    #[test]
    fn parses_empty_tensor_ranges() {
        let mut file = NamedTempFile::new().unwrap();
        let header = br#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        let parsed = read_safetensors_header(file.path()).unwrap();
        assert_eq!(parsed.tensors[0].byte_length, ByteLength(0));
    }
}
