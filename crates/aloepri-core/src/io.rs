use crate::{
    error::{CompilerError, Result},
    memory::MemoryBudget,
    types::{
        ByteLength, ByteOffset, ModelFingerprint, OutputTensorDescriptor, TensorDescriptor,
        TensorName,
    },
};

pub trait TensorReader: Send {
    fn len(&self) -> ByteLength;
    fn is_empty(&self) -> bool {
        self.len().0 == 0
    }
    fn read_bytes(&mut self, offset: ByteOffset, destination: &mut [u8]) -> Result<()>;
}

/// A bounded destination for exactly one output tensor.
///
/// The compiler creates a sink, hands the executor only a writable borrow, and
/// calls [`TensorSink::finish`] itself. A sink therefore owns the completion
/// contract: it counts what was actually written and refuses to finish unless
/// every planned byte arrived.
pub trait TensorSink {
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<ModelFingerprint>;
}

pub trait TensorWriter {
    fn begin_tensor(&mut self, output: &OutputTensorDescriptor)
    -> Result<Box<dyn TensorSink + '_>>;

    fn sync(&mut self) -> Result<()>;
}

pub trait OutputWriter: TensorWriter {
    fn hash_tensor(&mut self, name: &TensorName) -> Result<ModelFingerprint>;
    fn write_index(&mut self) -> Result<()>;
}

/// Copy `length` bytes from `reader` into `writer` in bounded chunks.
///
/// The chunk never exceeds `budget.available()` nor 4 MiB, so a tensor far
/// larger than the working buffer still streams without being materialized.
pub fn copy_bytes(
    reader: &mut dyn TensorReader,
    writer: &mut dyn FnMut(&[u8]) -> Result<()>,
    length: ByteLength,
    budget: &MemoryBudget,
) -> Result<ModelFingerprint> {
    let available = budget.available();
    if length.0 > 0 && available == 0 {
        return Err(CompilerError::MemoryLimitExceeded {
            requested: 1,
            available: 0,
        });
    }
    let chunk_size = if length.0 == 0 {
        1
    } else {
        length.0.min(4 * 1024 * 1024).min(available)
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

/// Copy a full source tensor identified by `source` into an output sink.
pub fn copy_source_to_sink(
    reader: &mut dyn TensorReader,
    sink: &mut dyn TensorSink,
    source: &TensorDescriptor,
    output: &OutputTensorDescriptor,
    budget: &MemoryBudget,
) -> Result<()> {
    if source.byte_length != output.byte_length {
        return Err(CompilerError::InvalidPlan {
            reason: format!(
                "copy of {} changes byte length from {} to {}",
                output.name, source.byte_length.0, output.byte_length.0
            ),
        });
    }
    let mut write = |bytes: &[u8]| sink.write_bytes(bytes);
    copy_bytes(reader, &mut write, output.byte_length, budget)?;
    Ok(())
}
