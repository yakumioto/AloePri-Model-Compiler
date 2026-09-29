use aloepri_core::backend::OutputLockGuard;
use aloepri_core::error::{CompilerError, Result, io_error};
use rustix::fs::{RenameFlags, renameat_with};
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

pub struct OutputLock {
    _file: File,
    path: PathBuf,
}

impl OutputLockGuard for OutputLock {}

impl OutputLock {
    pub fn acquire(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_owned();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|source| io_error(&path, source))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file, path }),
            Err(std::fs::TryLockError::WouldBlock) => Err(CompilerError::LockUnavailable { path }),
            Err(std::fs::TryLockError::Error(source)) => Err(io_error(path, source)),
        }
    }
}

impl Drop for OutputLock {
    fn drop(&mut self) {
        let _ = self._file.unlock();
        let _ = &self.path;
    }
}

pub fn publish_no_replace(staging: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        return Err(CompilerError::AlreadyExists {
            path: output.to_owned(),
        });
    }
    let source_parent = staging.parent().unwrap_or_else(|| Path::new("."));
    let output_parent = output.parent().unwrap_or_else(|| Path::new("."));
    let source_name = staging
        .file_name()
        .ok_or_else(|| CompilerError::Invariant("staging path has no file name".into()))?;
    let output_name = output
        .file_name()
        .ok_or_else(|| CompilerError::Invariant("output path has no file name".into()))?;
    let source_directory =
        File::open(source_parent).map_err(|source| io_error(source_parent, source))?;
    let output_directory =
        File::open(output_parent).map_err(|source| io_error(output_parent, source))?;
    renameat_with(
        &source_directory,
        source_name,
        &output_directory,
        output_name,
        RenameFlags::NOREPLACE,
    )
    .map_err(|source| io_error(output, std::io::Error::from(source)))?;
    sync_directory(output_parent)
}

pub fn sync_directory(path: &Path) -> Result<()> {
    let directory = File::open(path).map_err(|source| io_error(path, source))?;
    directory
        .sync_all()
        .map_err(|source| io_error(path, source))
}
