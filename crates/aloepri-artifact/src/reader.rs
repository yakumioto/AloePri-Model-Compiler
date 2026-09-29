use aloepri_core::{
    error::{CompilerError, Result, io_error},
    io::TensorReader,
    types::{ByteLength, ByteOffset},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

pub struct FileTensorReader {
    path: PathBuf,
    file: File,
    absolute_offset: u64,
    length: ByteLength,
}

impl FileTensorReader {
    pub fn open(path: &Path, absolute_offset: u64, length: ByteLength) -> Result<Self> {
        let file = File::open(path).map_err(|source| io_error(path, source))?;
        Ok(Self {
            path: path.to_owned(),
            file,
            absolute_offset,
            length,
        })
    }
}

impl TensorReader for FileTensorReader {
    fn len(&self) -> ByteLength {
        self.length
    }

    fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()> {
        let requested =
            u64::try_from(destination.len()).map_err(|_| CompilerError::ArithmeticOverflow {
                operation: "tensor read length",
            })?;
        let end = offset
            .0
            .checked_add(requested)
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "tensor read range",
            })?;
        if end > self.length.0 {
            return Err(CompilerError::OutputCorrupted {
                reason: format!(
                    "read range [{}, {end}) exceeds tensor length {}",
                    offset.0, self.length.0
                ),
            });
        }
        let absolute = self.absolute_offset.checked_add(offset.0).ok_or(
            CompilerError::ArithmeticOverflow {
                operation: "absolute tensor read offset",
            },
        )?;
        self.file
            .seek(SeekFrom::Start(absolute))
            .map_err(|source| io_error(&self.path, source))?;
        self.file
            .read_exact(destination)
            .map_err(|source| io_error(&self.path, source))
    }
}
