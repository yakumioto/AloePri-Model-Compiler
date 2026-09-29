use aloepri_core::{
    error::{CompilerError, Result, io_error},
    io::{OutputWriter, TensorReader, TensorWriter, copy_tensor},
    memory::MemoryBudget,
    plan::OutputLayout,
    types::{ByteLength, ModelFingerprint, TensorDescriptor, TensorName},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub struct StreamingWriter {
    root: PathBuf,
    layout: OutputLayout,
    files: BTreeMap<u32, File>,
}

impl StreamingWriter {
    pub fn create(root: impl AsRef<Path>, layout: OutputLayout) -> Result<Self> {
        let root = root.as_ref().to_owned();
        fs::create_dir_all(&root).map_err(|source| io_error(&root, source))?;
        let mut files = BTreeMap::new();
        for shard in &layout.shards {
            let path = root.join(&shard.filename);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|source| {
                    if source.kind() == std::io::ErrorKind::AlreadyExists {
                        CompilerError::AlreadyExists { path: path.clone() }
                    } else {
                        io_error(&path, source)
                    }
                })?;
            write_header(&mut file, &shard.header, shard.file_length, &path)?;
            files.insert(shard.id, file);
        }
        Ok(Self {
            root,
            layout,
            files,
        })
    }

    pub fn resume(root: impl AsRef<Path>, layout: OutputLayout) -> Result<Self> {
        let root = root.as_ref().to_owned();
        if !root.is_dir() {
            return Err(CompilerError::ResumeMismatch {
                reason: format!("staging artifact directory {} is missing", root.display()),
            });
        }
        let planned: BTreeSet<&str> = layout
            .shards
            .iter()
            .map(|shard| shard.filename.as_str())
            .collect();
        for entry in fs::read_dir(&root).map_err(|source| io_error(&root, source))? {
            let entry = entry.map_err(|source| io_error(&root, source))?;
            let path = entry.path();
            let is_weight_file = path
                .extension()
                .is_some_and(|extension| extension == "safetensors");
            let name = path.file_name().and_then(|value| value.to_str());
            if is_weight_file && !name.is_some_and(|value| planned.contains(value)) {
                return Err(CompilerError::ResumeMismatch {
                    reason: format!(
                        "{} is not part of the planned output layout",
                        path.display()
                    ),
                });
            }
        }
        let mut files = BTreeMap::new();
        for shard in &layout.shards {
            let path = root.join(&shard.filename);
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|source| io_error(&path, source))?;
            let length = file
                .metadata()
                .map_err(|source| io_error(&path, source))?
                .len();
            if length != shard.file_length.0 {
                return Err(CompilerError::ResumeMismatch {
                    reason: format!(
                        "{} has length {length}, expected {}",
                        path.display(),
                        shard.file_length.0
                    ),
                });
            }
            let mut prefix = [0_u8; 8];
            file.read_exact(&mut prefix)
                .map_err(|source| io_error(&path, source))?;
            let header_length = u64::from_le_bytes(prefix);
            if header_length != shard.header.len() as u64 {
                return Err(CompilerError::ResumeMismatch {
                    reason: format!("{} has an unexpected header length", path.display()),
                });
            }
            let mut existing_header = vec![0_u8; shard.header.len()];
            file.read_exact(&mut existing_header)
                .map_err(|source| io_error(&path, source))?;
            if existing_header != shard.header {
                return Err(CompilerError::ResumeMismatch {
                    reason: format!("{} has an unexpected output header", path.display()),
                });
            }
            files.insert(shard.id, file);
        }
        Ok(Self {
            root,
            layout,
            files,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn layout(&self) -> &OutputLayout {
        &self.layout
    }

    pub fn tensor_hash(&mut self, name: &TensorName) -> Result<ModelFingerprint> {
        let output = self
            .layout
            .tensor(name)
            .ok_or_else(|| CompilerError::MissingTensor {
                name: name.to_string(),
            })?;
        let shard = self
            .layout
            .shards
            .get(output.shard as usize)
            .ok_or_else(|| CompilerError::Invariant("output shard missing".into()))?;
        let path = self.root.join(&shard.filename);
        let absolute = 8_u64
            .checked_add(shard.header.len() as u64)
            .and_then(|value| value.checked_add(output.offset.0))
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "output tensor hash offset",
            })?;
        let mut file = File::open(&path).map_err(|source| io_error(&path, source))?;
        file.seek(SeekFrom::Start(absolute))
            .map_err(|source| io_error(&path, source))?;
        let mut remaining = output.byte_length.0;
        let mut hasher = blake3::Hasher::new();
        let buffer_size = if remaining == 0 {
            1
        } else {
            remaining.min(4 * 1024 * 1024) as usize
        };
        let mut buffer = vec![0_u8; buffer_size];
        while remaining > 0 {
            let size = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..size])
                .map_err(|source| io_error(&path, source))?;
            hasher.update(&buffer[..size]);
            remaining -= size as u64;
        }
        Ok(ModelFingerprint::from_digest(hasher.finalize()))
    }
}

impl TensorWriter for StreamingWriter {
    fn write_tensor(
        &mut self,
        descriptor: &TensorDescriptor,
        reader: &mut dyn TensorReader,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint> {
        let output =
            self.layout
                .tensor(&descriptor.name)
                .ok_or_else(|| CompilerError::MissingTensor {
                    name: descriptor.name.to_string(),
                })?;
        if descriptor.byte_length != output.byte_length
            || descriptor.dtype != output.dtype
            || descriptor.shape != output.shape
        {
            return Err(CompilerError::InvalidPlan {
                reason: format!("output metadata differs for {}", descriptor.name),
            });
        }
        if reader.len() != descriptor.byte_length {
            return Err(CompilerError::InvalidTensor {
                name: descriptor.name.to_string(),
                reason: format!(
                    "reader length {} differs from descriptor {}",
                    reader.len().0,
                    descriptor.byte_length.0
                ),
            });
        }
        let shard = self
            .layout
            .shards
            .get(output.shard as usize)
            .ok_or_else(|| CompilerError::Invariant("output shard missing".into()))?;
        let absolute = 8_u64
            .checked_add(shard.header.len() as u64)
            .and_then(|value| value.checked_add(output.offset.0))
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "output tensor write offset",
            })?;
        let file = self
            .files
            .get_mut(&shard.id)
            .ok_or_else(|| CompilerError::Invariant("output shard file is closed".into()))?;
        file.seek(SeekFrom::Start(absolute))
            .map_err(|source| io_error(self.root.join(&shard.filename), source))?;
        let path = self.root.join(&shard.filename);
        let mut append = |bytes: &[u8]| {
            file.write_all(bytes)
                .map_err(|source| io_error(&path, source))
        };
        copy_tensor(reader, &mut append, descriptor.byte_length, budget)
    }

    fn sync(&mut self) -> Result<()> {
        for (id, file) in &self.files {
            file.sync_all()
                .map_err(|source| io_error(self.root.join(format!("shard-{id}")), source))?;
        }
        Ok(())
    }
}

impl OutputWriter for StreamingWriter {
    fn hash_tensor(&mut self, name: &TensorName) -> Result<ModelFingerprint> {
        self.tensor_hash(name)
    }

    fn write_index(&mut self) -> Result<()> {
        if let Some(index) = &self.layout.index {
            let path = self.root.join("model.safetensors.index.json");
            let mut file = File::create(&path).map_err(|source| io_error(&path, source))?;
            file.write_all(index)
                .map_err(|source| io_error(&path, source))?;
            file.sync_all().map_err(|source| io_error(&path, source))?;
        }
        Ok(())
    }
}

fn write_header(
    file: &mut File,
    header: &[u8],
    file_length: ByteLength,
    path: &Path,
) -> Result<()> {
    file.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(|source| io_error(path, source))?;
    file.write_all(header)
        .map_err(|source| io_error(path, source))?;
    file.set_len(file_length.0)
        .map_err(|source| io_error(path, source))?;
    file.sync_all().map_err(|source| io_error(path, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aloepri_core::types::{ByteOffset, DType, ShardId, TensorLocation, TensorShape};
    use std::io::Cursor;
    use tempfile::tempdir;

    struct Reader(Cursor<Vec<u8>>);

    impl TensorReader for Reader {
        fn len(&self) -> ByteLength {
            ByteLength(self.0.get_ref().len() as u64)
        }
        fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()> {
            self.0.set_position(offset.0);
            self.0
                .read_exact(destination)
                .map_err(|source| io_error("reader", source))
        }
    }

    #[test]
    fn writer_copies_payload_without_whole_tensor_requirement() {
        let descriptor = TensorDescriptor {
            name: TensorName::try_from("x").unwrap(),
            shape: TensorShape::new(vec![3]),
            dtype: DType::U8,
            byte_length: ByteLength(3),
            location: TensorLocation {
                shard: ShardId(0),
                offset: ByteOffset(0),
                length: ByteLength(3),
            },
        };
        let layout =
            crate::layout::plan_output_layout(std::slice::from_ref(&descriptor), ByteLength(10))
                .unwrap();
        let directory = tempdir().unwrap();
        let mut writer = StreamingWriter::create(directory.path(), layout).unwrap();
        let hash = writer
            .write_tensor(
                &descriptor,
                &mut Reader(Cursor::new(vec![1, 2, 3])),
                &MemoryBudget::new(2),
            )
            .unwrap();
        writer.sync().unwrap();
        assert_eq!(hash, writer.tensor_hash(&descriptor.name).unwrap());
    }

    #[test]
    fn resume_rejects_weight_files_outside_the_planned_layout() {
        let descriptor = TensorDescriptor {
            name: TensorName::try_from("x").unwrap(),
            shape: TensorShape::new(vec![3]),
            dtype: DType::U8,
            byte_length: ByteLength(3),
            location: TensorLocation {
                shard: ShardId(0),
                offset: ByteOffset(0),
                length: ByteLength(3),
            },
        };
        let layout =
            crate::layout::plan_output_layout(std::slice::from_ref(&descriptor), ByteLength(3))
                .unwrap();
        let directory = tempdir().unwrap();
        drop(StreamingWriter::create(directory.path(), layout.clone()).unwrap());
        fs::write(directory.path().join("stray.safetensors"), b"junk").unwrap();
        assert!(StreamingWriter::resume(directory.path(), layout).is_err());
    }
}
