use crate::error::CompilerError;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TensorName(String);

impl TensorName {
    pub fn new(value: impl Into<String>) -> Result<Self, CompilerError> {
        let value = value.into();
        if value.is_empty() || value.as_bytes().contains(&0) {
            return Err(CompilerError::InvalidTensor {
                name: value,
                reason: "tensor name must be non-empty and contain no NUL".into(),
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TensorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TryFrom<String> for TensorName {
    type Error = CompilerError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for TensorName {
    type Error = CompilerError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TensorIndex(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ShardId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerIndex(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OperationId(pub u32);

#[derive(
    Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ByteOffset(pub u64);

impl ByteOffset {
    pub fn checked_add(self, value: ByteLength) -> Result<Self, CompilerError> {
        self.0
            .checked_add(value.0)
            .map(Self)
            .ok_or(CompilerError::ArithmeticOverflow {
                operation: "byte offset addition",
            })
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ByteLength(pub u64);

impl ByteLength {
    pub fn as_usize(self) -> Result<usize, CompilerError> {
        usize::try_from(self.0).map_err(|_| CompilerError::ArithmeticOverflow {
            operation: "byte length conversion",
        })
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ElementOffset(pub u64);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TensorShape(pub Vec<u64>);

impl TensorShape {
    pub fn new(values: Vec<u64>) -> Self {
        Self(values)
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }

    pub fn element_count(&self) -> Result<u64, CompilerError> {
        self.0.iter().try_fold(1_u64, |acc, value| {
            acc.checked_mul(*value)
                .ok_or(CompilerError::ArithmeticOverflow {
                    operation: "tensor element count",
                })
        })
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.0).finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    BOOL,
}

impl DType {
    pub fn from_safetensors(value: &str) -> Result<Self, CompilerError> {
        match value {
            "F64" => Ok(Self::F64),
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I64" => Ok(Self::I64),
            "I32" => Ok(Self::I32),
            "I16" => Ok(Self::I16),
            "I8" => Ok(Self::I8),
            "U64" => Ok(Self::U64),
            "U32" => Ok(Self::U32),
            "U16" => Ok(Self::U16),
            "U8" => Ok(Self::U8),
            "BOOL" => Ok(Self::BOOL),
            other => Err(CompilerError::UnsupportedDType {
                dtype: other.to_owned(),
            }),
        }
    }

    pub fn as_safetensors(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U64 => "U64",
            Self::U32 => "U32",
            Self::U16 => "U16",
            Self::U8 => "U8",
            Self::BOOL => "BOOL",
        }
    }

    pub fn byte_width(self) -> u64 {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::BOOL => 1,
        }
    }

    pub fn alignment(self) -> u64 {
        self.byte_width()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OutputDType {
    Preserve,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorLocation {
    pub shard: ShardId,
    pub offset: ByteOffset,
    pub length: ByteLength,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub name: TensorName,
    pub shape: TensorShape,
    pub dtype: DType,
    pub byte_length: ByteLength,
    pub location: TensorLocation,
}

impl TensorDescriptor {
    pub fn expected_byte_length(&self) -> Result<ByteLength, CompilerError> {
        let elements = self.shape.element_count()?;
        let bytes = elements.checked_mul(self.dtype.byte_width()).ok_or(
            CompilerError::ArithmeticOverflow {
                operation: "tensor byte length",
            },
        )?;
        Ok(ByteLength(bytes))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelFingerprint([u8; 32]);

impl ModelFingerprint {
    pub fn from_digest(digest: blake3::Hash) -> Self {
        Self(*digest.as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_hex(value: &str) -> Result<Self, CompilerError> {
        if value.len() != 64 {
            return Err(CompilerError::Invariant(
                "model fingerprint must contain 64 hexadecimal characters".into(),
            ));
        }
        let mut bytes = [0_u8; 32];
        for (index, chunk) in value.as_bytes().chunks(2).enumerate() {
            let high = hex_digit(chunk[0]).ok_or_else(|| {
                CompilerError::Invariant("model fingerprint contains non-hexadecimal data".into())
            })?;
            let low = hex_digit(chunk[1]).ok_or_else(|| {
                CompilerError::Invariant("model fingerprint contains non-hexadecimal data".into())
            })?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl fmt::Display for ModelFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_has_one_element_and_zero_dimension_is_empty() {
        assert_eq!(TensorShape::new(vec![]).element_count().unwrap(), 1);
        assert_eq!(TensorShape::new(vec![2, 0, 3]).element_count().unwrap(), 0);
    }

    #[test]
    fn dtype_sizes_are_byte_aligned() {
        assert_eq!(DType::BF16.byte_width(), 2);
        assert_eq!(DType::F32.byte_width(), 4);
    }

    #[test]
    fn byte_offset_checks_overflow() {
        assert!(ByteOffset(u64::MAX).checked_add(ByteLength(1)).is_err());
    }
}
