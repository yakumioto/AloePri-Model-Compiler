use crate::{
    header::{ShardHeader, read_safetensors_header},
    reader::FileTensorReader,
};
use aloepri_core::{
    backend::ShardSummary,
    error::{CompilerError, Result, io_error, json_error},
    io::TensorReader,
    model::{ModelArtifact, ModelSpec},
    types::{ByteOffset, ModelFingerprint, ShardId, TensorDescriptor, TensorLocation, TensorName},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

const SIDECAR_NAMES: &[&str] = &[
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
];

#[derive(Clone, Debug, Serialize)]
pub struct ShardInfo {
    pub id: ShardId,
    pub filename: String,
    pub file_length: u64,
    pub payload_length: u64,
}

#[derive(Debug)]
pub struct HfArtifact {
    root: PathBuf,
    config_bytes: Vec<u8>,
    config: Value,
    spec: ModelSpec,
    shards: Vec<ShardInfo>,
    shard_headers: Vec<ShardHeader>,
}

#[derive(Deserialize)]
struct IndexFile {
    #[serde(default)]
    metadata: Option<IndexMetadata>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct IndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

impl HfArtifact {
    pub fn discover(root: impl AsRef<Path>) -> Result<Self> {
        Self::open(root)
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|source| io_error(root.as_ref(), source))?;
        if !root.is_dir() {
            return Err(CompilerError::InvalidArtifact {
                path: root,
                reason: "artifact root is not a directory".into(),
            });
        }
        let config_path = validate_relative_path(&root, Path::new("config.json"))?;
        let config_bytes =
            fs::read(&config_path).map_err(|source| io_error(&config_path, source))?;
        crate::header::reject_duplicate_keys(&config_bytes).map_err(|reason| {
            CompilerError::InvalidArtifact {
                path: config_path.clone(),
                reason: format!("invalid or duplicate config JSON: {reason}"),
            }
        })?;
        let config: Value = serde_json::from_slice(&config_bytes)
            .map_err(|source| json_error(&config_path, source))?;
        let single_candidate = root.join("model.safetensors");
        let index_candidate = root.join("model.safetensors.index.json");
        let single_exists = single_candidate.is_file();
        let index_exists = index_candidate.is_file();
        let single = if single_exists {
            validate_relative_path(&root, Path::new("model.safetensors"))?
        } else {
            single_candidate
        };
        let index = if index_exists {
            validate_relative_path(&root, Path::new("model.safetensors.index.json"))?
        } else {
            index_candidate
        };
        if single_exists && index_exists {
            return Err(CompilerError::InvalidArtifact {
                path: root.clone(),
                reason: "both single-file and indexed safetensors layouts are present".into(),
            });
        }
        let indexed_map = if index_exists {
            Some(Self::read_index(&index)?)
        } else {
            None
        };
        let shard_paths = if let Some(index_file) = indexed_map.as_ref() {
            let mut paths = BTreeSet::new();
            for shard in index_file.weight_map.values() {
                paths.insert(validate_relative_path(&root, Path::new(shard))?);
            }
            paths.into_iter().collect::<Vec<_>>()
        } else if single_exists {
            vec![single]
        } else {
            return Err(CompilerError::InvalidArtifact {
                path: root,
                reason: "expected model.safetensors or model.safetensors.index.json".into(),
            });
        };
        let mut shard_headers = Vec::new();
        let mut shards = Vec::new();
        for (index, path) in shard_paths.iter().enumerate() {
            if !path.is_file() {
                return Err(CompilerError::InvalidArtifact {
                    path: path.clone(),
                    reason: "referenced safetensors shard does not exist".into(),
                });
            }
            let header = read_safetensors_header(path)?;
            let payload_length = header.file_length - header.data_start;
            shards.push(ShardInfo {
                id: ShardId(index as u32),
                filename: path
                    .strip_prefix(&root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .into_owned(),
                file_length: header.file_length,
                payload_length,
            });
            shard_headers.push(header);
        }
        let mut tensors = BTreeMap::new();
        for (shard_index, header) in shard_headers.iter().enumerate() {
            for tensor in &header.tensors {
                if tensors.contains_key(&tensor.name) {
                    return Err(CompilerError::InvalidArtifact {
                        path: header.path.clone(),
                        reason: format!("tensor {} appears in multiple shards", tensor.name),
                    });
                }
                let offset = header
                    .data_start
                    .checked_add(tensor.relative_offset)
                    .ok_or(CompilerError::ArithmeticOverflow {
                        operation: "tensor absolute offset",
                    })?;
                tensors.insert(
                    tensor.name.clone(),
                    TensorDescriptor {
                        name: tensor.name.clone(),
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                        byte_length: tensor.byte_length,
                        location: TensorLocation {
                            shard: ShardId(shard_index as u32),
                            offset: ByteOffset(offset),
                            length: tensor.byte_length,
                        },
                    },
                );
            }
        }
        if let Some(index_file) = indexed_map.as_ref() {
            let expected: BTreeSet<TensorName> = index_file
                .weight_map
                .keys()
                .map(|name| TensorName::try_from(name.as_str()))
                .collect::<Result<_>>()?;
            let actual: BTreeSet<TensorName> = tensors.keys().cloned().collect();
            if expected != actual {
                return Err(CompilerError::InvalidArtifact {
                    path: index,
                    reason: "weight_map tensor set differs from physical shard tensor set".into(),
                });
            }
            for (name, shard_name) in &index_file.weight_map {
                let tensor_name = TensorName::try_from(name.as_str())?;
                let descriptor = tensors.get(&tensor_name).expect("validated tensor set");
                let actual_shard = &shards[descriptor.location.shard.0 as usize].filename;
                if actual_shard != shard_name {
                    return Err(CompilerError::InvalidArtifact {
                        path: root.clone(),
                        reason: format!(
                            "index maps {name} to {shard_name}, physical tensor is in {actual_shard}"
                        ),
                    });
                }
            }
            if let Some(expected_size) = index_file
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.total_size)
            {
                let actual_size = tensors
                    .values()
                    .map(|tensor| tensor.byte_length.0)
                    .sum::<u64>();
                if expected_size != actual_size {
                    return Err(CompilerError::InvalidArtifact {
                        path: index,
                        reason: format!(
                            "index total_size {expected_size} differs from payload {actual_size}"
                        ),
                    });
                }
            }
        }
        let format = shard_headers
            .iter()
            .filter_map(|header| header.metadata.get("format"))
            .collect::<Vec<_>>();
        if format.windows(2).any(|values| values[0] != values[1]) {
            return Err(CompilerError::InvalidArtifact {
                path: root.clone(),
                reason: "shards contain conflicting format metadata".into(),
            });
        }
        let architecture = config
            .get("model_type")
            .and_then(Value::as_str)
            .or_else(|| {
                config
                    .get("architectures")
                    .and_then(Value::as_array)
                    .and_then(|values| values.first())
                    .and_then(Value::as_str)
            })
            .unwrap_or("unknown")
            .to_owned();
        let mut aliases = BTreeMap::new();
        if config.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true)
            && !tensors.contains_key(&TensorName::try_from("lm_head.weight")?)
            && tensors.contains_key(&TensorName::try_from("model.embed_tokens.weight")?)
        {
            aliases.insert("lm_head.weight".into(), "model.embed_tokens.weight".into());
        }
        let spec = ModelSpec {
            architecture,
            config: config.clone(),
            tensors: tensors.into_values().collect(),
            aliases,
        };
        Ok(Self {
            root,
            config_bytes,
            config,
            spec,
            shards,
            shard_headers,
        })
    }

    fn read_index(path: &Path) -> Result<IndexFile> {
        let bytes = fs::read(path).map_err(|source| io_error(path, source))?;
        crate::header::reject_duplicate_keys(&bytes).map_err(|reason| {
            CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: format!("invalid or duplicate index JSON: {reason}"),
            }
        })?;
        serde_json::from_slice(&bytes).map_err(|source| json_error(path, source))
    }

    pub fn shards(&self) -> &[ShardInfo] {
        &self.shards
    }

    pub fn tensor(&self, name: &TensorName) -> Result<Box<dyn TensorReader>> {
        self.tensor_reader(name)
    }

    pub fn sidecar_bytes(&self) -> Result<BTreeMap<String, Vec<u8>>> {
        let mut result = BTreeMap::new();
        for name in SIDECAR_NAMES {
            let path = self.root.join(name);
            if path.is_file() {
                validate_relative_path(&self.root, Path::new(name))?;
                result.insert(
                    (*name).into(),
                    fs::read(&path).map_err(|source| io_error(path, source))?,
                );
            }
        }
        Ok(result)
    }

    pub fn total_payload_bytes(&self) -> u64 {
        self.spec
            .tensors
            .iter()
            .map(|tensor| tensor.byte_length.0)
            .sum()
    }
}

impl ModelArtifact for HfArtifact {
    fn root(&self) -> &Path {
        &self.root
    }

    fn config_bytes(&self) -> &[u8] {
        &self.config_bytes
    }

    fn config(&self) -> &Value {
        &self.config
    }

    fn model_spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn sidecars(&self) -> Result<BTreeMap<String, Vec<u8>>> {
        self.sidecar_bytes()
    }

    fn shards(&self) -> Vec<ShardSummary> {
        self.shards
            .iter()
            .map(|shard| ShardSummary {
                filename: shard.filename.clone(),
                file_length: shard.file_length,
                payload_length: shard.payload_length,
            })
            .collect()
    }

    fn tensor_reader(&self, name: &TensorName) -> Result<Box<dyn TensorReader>> {
        let descriptor = self
            .spec
            .tensor(name)
            .ok_or_else(|| CompilerError::MissingTensor {
                name: name.to_string(),
            })?;
        let header = self
            .shard_headers
            .get(descriptor.location.shard.0 as usize)
            .ok_or_else(|| CompilerError::InvalidArtifact {
                path: self.root.clone(),
                reason: format!("missing shard {}", descriptor.location.shard.0),
            })?;
        Ok(Box::new(FileTensorReader::open(
            &header.path,
            descriptor.location.offset.0,
            descriptor.byte_length,
        )?))
    }

    fn fingerprint(&self) -> Result<ModelFingerprint> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aloepri-artifact-fingerprint-v1");
        hash_bytes(&mut hasher, b"config", &self.config_bytes);
        for (name, bytes) in self.sidecar_bytes()? {
            hash_bytes(&mut hasher, name.as_bytes(), &bytes);
        }
        for descriptor in &self.spec.tensors {
            hash_bytes(
                &mut hasher,
                b"tensor-name",
                descriptor.name.as_str().as_bytes(),
            );
            hash_bytes(
                &mut hasher,
                b"tensor-dtype",
                descriptor.dtype.as_safetensors().as_bytes(),
            );
            let shape = serde_json::to_vec(&descriptor.shape)
                .map_err(|error| CompilerError::Invariant(error.to_string()))?;
            hash_bytes(&mut hasher, b"tensor-shape", &shape);
            hash_bytes(
                &mut hasher,
                b"tensor-length",
                &descriptor.byte_length.0.to_le_bytes(),
            );
            let mut reader = self.tensor_reader(&descriptor.name)?;
            let mut remaining = descriptor.byte_length.0;
            let mut offset = 0_u64;
            let buffer_size = if remaining == 0 {
                1
            } else {
                remaining.min(4 * 1024 * 1024) as usize
            };
            let mut buffer = vec![0_u8; buffer_size];
            while remaining > 0 {
                let size = remaining.min(buffer.len() as u64) as usize;
                reader.read_bytes(ByteOffset(offset), &mut buffer[..size])?;
                hasher.update(&buffer[..size]);
                offset += size as u64;
                remaining -= size as u64;
            }
        }
        Ok(ModelFingerprint::from_digest(hasher.finalize()))
    }
}

fn hash_bytes(hasher: &mut blake3::Hasher, label: &[u8], bytes: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn validate_relative_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(CompilerError::PathEscape {
            path: relative.to_owned(),
        });
    }
    let candidate = root.join(relative);
    let canonical = candidate
        .canonicalize()
        .map_err(|source| io_error(&candidate, source))?;
    if !canonical.starts_with(root) {
        return Err(CompilerError::PathEscape {
            path: relative.to_owned(),
        });
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_model(root: &Path) {
        fs::write(
            root.join("config.json"),
            br#"{"model_type":"llama","tie_word_embeddings":true}"#,
        )
        .unwrap();
        let header = br#"{"x":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#;
        let mut file = fs::File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[1, 2, 3]).unwrap();
    }

    fn write_shard(path: &Path, name: &str, bytes: &[u8]) {
        let mut header = serde_json::Map::new();
        header.insert(
            name.into(),
            serde_json::json!({
                "dtype": "U8",
                "shape": [bytes.len()],
                "data_offsets": [0, bytes.len()],
            }),
        );
        let mut header_bytes = serde_json::to_vec(&header).unwrap();
        header_bytes.resize(
            header_bytes.len() + ((8 - header_bytes.len() % 8) % 8),
            b' ',
        );
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(bytes).unwrap();
    }

    #[test]
    fn discovers_single_file() {
        let directory = tempdir().unwrap();
        write_model(directory.path());
        let artifact = HfArtifact::open(directory.path()).unwrap();
        assert_eq!(artifact.tensors().len(), 1);
        assert_eq!(artifact.total_payload_bytes(), 3);
    }

    #[test]
    fn discovers_indexed_shards_and_checks_weight_map() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            br#"{"model_type":"llama"}"#,
        )
        .unwrap();
        write_shard(&directory.path().join("part-a.safetensors"), "a", &[1, 2]);
        write_shard(&directory.path().join("part-b.safetensors"), "b", &[3]);
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 3},
                "weight_map": {"a": "part-a.safetensors", "b": "part-b.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let artifact = HfArtifact::open(directory.path()).unwrap();
        assert_eq!(artifact.shards().len(), 2);
        assert_eq!(artifact.tensors().len(), 2);
    }

    fn write_index(root: &Path, weight_map: serde_json::Value) {
        fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn rejects_index_pointing_at_the_wrong_shard() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            br#"{"model_type":"llama"}"#,
        )
        .unwrap();
        write_shard(&directory.path().join("part-a.safetensors"), "a", &[1, 2]);
        write_shard(&directory.path().join("part-b.safetensors"), "b", &[3]);
        write_index(
            directory.path(),
            serde_json::json!({"a": "part-b.safetensors", "b": "part-b.safetensors"}),
        );
        assert!(HfArtifact::open(directory.path()).is_err());
    }

    #[test]
    fn rejects_index_missing_a_tensor() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            br#"{"model_type":"llama"}"#,
        )
        .unwrap();
        write_two_tensors_shard(&directory.path().join("part-a.safetensors"));
        write_index(
            directory.path(),
            serde_json::json!({"a": "part-a.safetensors"}),
        );
        assert!(HfArtifact::open(directory.path()).is_err());
    }

    fn write_two_tensors_shard(path: &Path) {
        let mut header = serde_json::Map::new();
        header.insert(
            "a".into(),
            serde_json::json!({"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}),
        );
        header.insert(
            "b".into(),
            serde_json::json!({"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}),
        );
        let mut header_bytes = serde_json::to_vec(&header).unwrap();
        header_bytes.resize(
            header_bytes.len() + ((8 - header_bytes.len() % 8) % 8),
            b' ',
        );
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&[1, 2]).unwrap();
    }

    #[test]
    fn rejects_same_tensor_in_two_shards() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            br#"{"model_type":"llama"}"#,
        )
        .unwrap();
        write_shard(&directory.path().join("part-a.safetensors"), "a", &[1]);
        write_shard(&directory.path().join("part-b.safetensors"), "a", &[2]);
        write_index(
            directory.path(),
            serde_json::json!({"a": "part-a.safetensors", "b": "part-b.safetensors"}),
        );
        assert!(HfArtifact::open(directory.path()).is_err());
    }
}
