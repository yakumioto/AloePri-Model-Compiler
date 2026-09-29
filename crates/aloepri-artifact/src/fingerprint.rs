use aloepri_core::{
    error::Result,
    io::TensorReader,
    types::{ByteOffset, ModelFingerprint, TensorDescriptor},
};

pub fn tensor_fingerprint(
    reader: &mut dyn TensorReader,
    descriptor: &TensorDescriptor,
) -> Result<ModelFingerprint> {
    let mut hasher = blake3::Hasher::new();
    let mut remaining = descriptor.byte_length.0;
    let mut offset = 0_u64;
    let chunk = if remaining == 0 {
        1
    } else {
        remaining.min(4 * 1024 * 1024) as usize
    };
    let mut buffer = vec![0_u8; chunk];
    while remaining > 0 {
        let size = remaining.min(buffer.len() as u64) as usize;
        reader.read_bytes(ByteOffset(offset), &mut buffer[..size])?;
        hasher.update(&buffer[..size]);
        offset += size as u64;
        remaining -= size as u64;
    }
    Ok(ModelFingerprint::from_digest(hasher.finalize()))
}
