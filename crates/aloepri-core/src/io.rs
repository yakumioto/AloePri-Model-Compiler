use crate::{
    error::{CompilerError, Result},
    memory::MemoryBudget,
    types::{ByteLength, ByteOffset, ModelFingerprint, TensorDescriptor, TensorName},
};

pub trait TensorReader: Send {
    fn len(&self) -> ByteLength;
    fn is_empty(&self) -> bool {
        self.len().0 == 0
    }
    fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()>;
}

pub trait TensorWriter {
    fn write_tensor(
        &mut self,
        descriptor: &TensorDescriptor,
        reader: &mut dyn TensorReader,
        budget: &MemoryBudget,
    ) -> Result<ModelFingerprint>;

    fn sync(&mut self) -> Result<()>;
}

pub trait OutputWriter: TensorWriter {
    fn hash_tensor(&mut self, name: &TensorName) -> Result<ModelFingerprint>;
    fn write_index(&mut self) -> Result<()>;
}

pub fn copy_tensor(
    reader: &mut dyn TensorReader,
    writer: &mut dyn FnMut(&[u8]) -> Result<()>,
    length: ByteLength,
    budget: &MemoryBudget,
) -> Result<ModelFingerprint> {
    if length.0 > 0 && budget.limit() == 0 {
        return Err(CompilerError::MemoryLimitExceeded {
            requested: 1,
            available: 0,
        });
    }
    let chunk_size = if length.0 == 0 {
        1
    } else {
        length.0.min(4 * 1024 * 1024).min(budget.limit())
    };
    let reservation = budget.reserve(chunk_size)?;
    let mut buffer = vec![0_u8; chunk_size as usize];
    let mut hasher = blake3::Hasher::new();
    let mut offset = 0_u64;
    while offset < length.0 {
        let remaining = length.0 - offset;
        let current = remaining.min(chunk_size) as usize;
        reader.read_bytes(ByteOffset(offset), &mut buffer[..current])?;
        writer(&buffer[..current])?;
        hasher.update(&buffer[..current]);
        offset += current as u64;
    }
    drop(reservation);
    Ok(ModelFingerprint::from_digest(hasher.finalize()))
}
