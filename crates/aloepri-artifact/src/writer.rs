use aloepri_core::{
    error::{CompilerError, Result, io_error},
    io::{OutputWriter, TensorSink, TensorWriter},
    plan::OutputLayout,
    types::{ByteLength, ModelFingerprint, OutputTensorDescriptor, TensorName},
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
    fn begin_tensor(
        &mut self,
        output: &OutputTensorDescriptor,
    ) -> Result<Box<dyn TensorSink + '_>> {
        let planned =
            self.layout
                .tensor(&output.name)
                .ok_or_else(|| CompilerError::MissingTensor {
                    name: output.name.to_string(),
                })?;
        if planned.shape != output.shape
            || planned.dtype != output.dtype
            || planned.byte_length != output.byte_length
        {
            return Err(CompilerError::InvalidPlan {
                reason: format!("output metadata differs for {}", output.name),
            });
        }
        let shard = self
            .layout
            .shards
            .get(planned.shard as usize)
            .ok_or_else(|| CompilerError::Invariant("output shard missing".into()))?;
        let absolute = 8_u64
            .checked_add(shard.header.len() as u64)
            .and_then(|value| value.checked_add(planned.offset.0))
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "output tensor write offset",
            })?;
        let path = self.root.join(&shard.filename);
        let file = self
            .files
            .get_mut(&shard.id)
            .ok_or_else(|| CompilerError::Invariant("output shard file is closed".into()))?;
        file.seek(SeekFrom::Start(absolute))
            .map_err(|source| io_error(&path, source))?;
        Ok(Box::new(ShardSink {
            file,
            path,
            expected: planned.byte_length,
            written: 0,
            hasher: blake3::Hasher::new(),
            failed: false,
        }))
    }

    fn sync(&mut self) -> Result<()> {
        for (id, file) in &self.files {
            file.sync_all()
                .map_err(|source| io_error(self.root.join(format!("shard-{id}")), source))?;
        }
        Ok(())
    }
}

/// A bounded sink over one output tensor's byte range.
///
/// It counts what was actually written, hashes exactly those bytes, and refuses
/// to finish short. An overlong write is rejected as a whole before any byte
/// reaches the file, so it can never spill into a neighbouring tensor.
struct ShardSink<'a> {
    file: &'a mut File,
    path: PathBuf,
    expected: ByteLength,
    written: u64,
    hasher: blake3::Hasher,
    failed: bool,
}

impl TensorSink for ShardSink<'_> {
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if self.failed {
            return Err(CompilerError::OutputCorrupted {
                reason: format!("{}: sink is in a failed state", self.path.display()),
            });
        }
        let length = u64::try_from(bytes.len()).map_err(|_| CompilerError::ArithmeticOverflow {
            operation: "output write length",
        })?;
        let end = self
            .written
            .checked_add(length)
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "output write range",
            })?;
        if end > self.expected.0 {
            self.failed = true;
            return Err(CompilerError::OutputCorrupted {
                reason: format!(
                    "{}: write of {length} bytes would overrun the output tensor by {} bytes",
                    self.path.display(),
                    end - self.expected.0
                ),
            });
        }
        if let Err(source) = self.file.write_all(bytes) {
            self.failed = true;
            return Err(io_error(&self.path, source));
        }
        self.hasher.update(bytes);
        self.written = end;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<ModelFingerprint> {
        if self.failed {
            return Err(CompilerError::OutputCorrupted {
                reason: format!("{}: sink failed before finishing", self.path.display()),
            });
        }
        if self.written != self.expected.0 {
            return Err(CompilerError::OutputCorrupted {
                reason: format!(
                    "{}: underwrite, wrote {} of {} bytes",
                    self.path.display(),
                    self.written,
                    self.expected.0
                ),
            });
        }
        Ok(ModelFingerprint::from_digest(self.hasher.finalize()))
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
    use aloepri_core::types::{DType, TensorName, TensorShape};
    use tempfile::tempdir;

    fn descriptor(name: &str, length: u64) -> OutputTensorDescriptor {
        OutputTensorDescriptor {
            name: TensorName::try_from(name).unwrap(),
            shape: TensorShape::new(vec![length]),
            dtype: DType::U8,
            byte_length: ByteLength(length),
        }
    }

    fn writer_with(descriptor: &OutputTensorDescriptor) -> (tempfile::TempDir, StreamingWriter) {
        let layout =
            crate::layout::plan_output_layout(std::slice::from_ref(descriptor), ByteLength(1024))
                .unwrap();
        let directory = tempdir().unwrap();
        let writer = StreamingWriter::create(directory.path(), layout).unwrap();
        (directory, writer)
    }

    #[test]
    fn sink_writes_exact_bytes_and_reports_their_digest() {
        let descriptor = descriptor("x", 3);
        let (_directory, mut writer) = writer_with(&descriptor);
        let digest = {
            let mut sink = writer.begin_tensor(&descriptor).unwrap();
            sink.write_bytes(&[1, 2, 3]).unwrap();
            sink.finish().unwrap()
        };
        writer.sync().unwrap();
        assert_eq!(digest, writer.tensor_hash(&descriptor.name).unwrap());
    }

    #[test]
    fn sink_rejects_underwrite_and_overwrite() {
        let descriptor = descriptor("x", 3);
        let (_directory, mut writer) = writer_with(&descriptor);
        {
            let mut sink = writer.begin_tensor(&descriptor).unwrap();
            sink.write_bytes(&[1, 2]).unwrap();
            assert!(sink.finish().is_err());
        }
        let (_directory, mut writer) = writer_with(&descriptor);
        {
            let mut sink = writer.begin_tensor(&descriptor).unwrap();
            assert!(sink.write_bytes(&[1, 2, 3, 4]).is_err());
            assert!(sink.write_bytes(&[1]).is_err());
        }
    }

    #[test]
    fn resume_rejects_weight_files_outside_the_planned_layout() {
        let descriptor = descriptor("x", 3);
        let layout =
            crate::layout::plan_output_layout(std::slice::from_ref(&descriptor), ByteLength(3))
                .unwrap();
        let directory = tempdir().unwrap();
        drop(StreamingWriter::create(directory.path(), layout.clone()).unwrap());
        fs::write(directory.path().join("stray.safetensors"), b"junk").unwrap();
        assert!(StreamingWriter::resume(directory.path(), layout).is_err());
    }
}
