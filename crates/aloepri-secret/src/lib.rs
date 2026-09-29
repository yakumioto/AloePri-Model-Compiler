use aloepri_core::{
    error::{CompilerError, Result, io_error, json_error},
    plan::{MethodContract, SecretBinding},
    types::ModelFingerprint,
};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fmt,
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
};

pub const CLIENT_SECRET_VERSION: u32 = 1;
const COMMITMENT_DOMAIN: &[u8] = b"aloepri-client-secret-v1";
const MAX_SECRET_FILE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientSecret {
    pub version: u32,
    pub method: MethodContract,
    pub secret_id: String,
    pub vocab_size: u64,
    pub source_fingerprint: String,
    pub binding_nonce: String,
    pub token_permutation: Vec<u32>,
    pub inverse_token_permutation: Vec<u32>,
}

impl fmt::Debug for ClientSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClientSecret")
            .field("version", &self.version)
            .field("method", &self.method)
            .field("secret_id", &self.secret_id)
            .field("vocab_size", &self.vocab_size)
            .field("source_fingerprint", &self.source_fingerprint)
            .finish_non_exhaustive()
    }
}

impl ClientSecret {
    pub fn generate(source_fingerprint: ModelFingerprint, vocab_size: u64) -> Result<Self> {
        let length = checked_vocab_len(vocab_size)?;
        let mut permutation: Vec<u32> = (0..length as u32).collect();
        loop {
            for index in (1..length).rev() {
                let bound =
                    u64::try_from(index + 1).map_err(|_| CompilerError::ArithmeticOverflow {
                        operation: "secret permutation index",
                    })?;
                let swap = random_below(bound)? as usize;
                permutation.swap(index, swap);
            }
            if permutation
                .iter()
                .enumerate()
                .any(|(index, value)| index as u32 != *value)
            {
                break;
            }
        }
        let mut nonce = [0_u8; 32];
        getrandom::fill(&mut nonce)
            .map_err(|error| CompilerError::Invariant(format!("OS randomness failed: {error}")))?;
        Self::from_components(source_fingerprint, vocab_size, nonce, permutation)
    }

    pub fn from_components(
        source_fingerprint: ModelFingerprint,
        vocab_size: u64,
        nonce: [u8; 32],
        token_permutation: Vec<u32>,
    ) -> Result<Self> {
        checked_vocab_len(vocab_size)?;
        let inverse = inverse_permutation(vocab_size, &token_permutation)?;
        let mut secret = Self {
            version: CLIENT_SECRET_VERSION,
            method: MethodContract::aloepri_token(),
            secret_id: String::new(),
            vocab_size,
            source_fingerprint: source_fingerprint.to_string(),
            binding_nonce: hex_encode(&nonce),
            token_permutation,
            inverse_token_permutation: inverse,
        };
        secret.secret_id = secret.compute_secret_id()?;
        secret.validate()?;
        Ok(secret)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path).map_err(|source| io_error(path, source))?;
        if metadata.len() > MAX_SECRET_FILE_BYTES {
            return Err(CompilerError::InvalidArtifact {
                path: path.to_owned(),
                reason: "client secret file is too large".into(),
            });
        }
        let bytes = fs::read(path).map_err(|source| io_error(path, source))?;
        reject_duplicate_keys(&bytes).map_err(|reason| CompilerError::InvalidArtifact {
            path: path.to_owned(),
            reason: format!("invalid or duplicate client secret JSON: {reason}"),
        })?;
        let raw: Value =
            serde_json::from_slice(&bytes).map_err(|source| json_error(path, source))?;
        validate_method_object(&raw)?;
        let secret: Self =
            serde_json::from_slice(&bytes).map_err(|source| json_error(path, source))?;
        secret.validate()?;
        Ok(secret)
    }

    pub fn write_new(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| CompilerError::Invariant(error.to_string()))?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(path)
            .map_err(|source| io_error(path, source))?;
        file.write_all(&bytes)
            .map_err(|source| io_error(path, source))?;
        file.sync_all().map_err(|source| io_error(path, source))?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let directory = File::open(parent).map_err(|source| io_error(parent, source))?;
        directory
            .sync_all()
            .map_err(|source| io_error(parent, source))?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != CLIENT_SECRET_VERSION {
            return Err(CompilerError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.method != MethodContract::aloepri_token() {
            return Err(CompilerError::Unsupported(
                "client secret method must be aloepri-token/0.1".into(),
            ));
        }
        checked_vocab_len(self.vocab_size)?;
        let source = ModelFingerprint::from_hex(&self.source_fingerprint).map_err(|_| {
            CompilerError::Invariant("client secret source fingerprint is not valid hex".into())
        })?;
        if source.to_string() != self.source_fingerprint {
            return Err(CompilerError::Invariant(
                "client secret source fingerprint must be lowercase hex".into(),
            ));
        }
        let nonce = decode_hex_32(&self.binding_nonce, "binding nonce")?;
        if self
            .token_permutation
            .iter()
            .enumerate()
            .all(|(index, value)| *value == index as u32)
        {
            return Err(CompilerError::Invariant(
                "client secret permutation must not be the identity".into(),
            ));
        }
        let expected_inverse = inverse_permutation(self.vocab_size, &self.token_permutation)?;
        if expected_inverse != self.inverse_token_permutation {
            return Err(CompilerError::Invariant(
                "client secret inverse permutation is inconsistent".into(),
            ));
        }
        let expected_id = self.compute_secret_id_with(source, nonce)?;
        if self.secret_id != expected_id {
            return Err(CompilerError::Invariant(
                "client secret commitment does not match its contents".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_for_model(
        &self,
        source_fingerprint: ModelFingerprint,
        vocab_size: u64,
    ) -> Result<()> {
        self.validate()?;
        if self.source_fingerprint != source_fingerprint.to_string()
            || self.vocab_size != vocab_size
        {
            return Err(CompilerError::InvalidPlan {
                reason: "client secret is bound to a different source or vocabulary".into(),
            });
        }
        Ok(())
    }

    pub fn binding(&self) -> Result<SecretBinding> {
        self.validate()?;
        Ok(SecretBinding {
            secret_id: self.secret_id.clone(),
            source_fingerprint: ModelFingerprint::from_hex(&self.source_fingerprint)?,
            vocab_size: self.vocab_size,
        })
    }

    pub fn permutation(&self) -> &[u32] {
        &self.token_permutation
    }

    pub fn inverse_permutation(&self) -> &[u32] {
        &self.inverse_token_permutation
    }

    fn compute_secret_id(&self) -> Result<String> {
        let source = ModelFingerprint::from_hex(&self.source_fingerprint)?;
        let nonce = decode_hex_32(&self.binding_nonce, "binding nonce")?;
        self.compute_secret_id_with(source, nonce)
    }

    fn compute_secret_id_with(&self, source: ModelFingerprint, nonce: [u8; 32]) -> Result<String> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMMITMENT_DOMAIN);
        hasher.update(&self.version.to_le_bytes());
        hash_string(&mut hasher, &self.method.id)?;
        hash_string(&mut hasher, &self.method.version)?;
        hasher.update(source.as_bytes());
        hasher.update(&self.vocab_size.to_le_bytes());
        hasher.update(&nonce);
        for value in &self.token_permutation {
            hasher.update(&value.to_le_bytes());
        }
        for value in &self.inverse_token_permutation {
            hasher.update(&value.to_le_bytes());
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

pub fn validate_v0_1_boundary(config: &aloepri_core::TransformConfig) -> Result<()> {
    if config.method != MethodContract::identity()
        && config.method != MethodContract::aloepri_token()
    {
        return Err(CompilerError::Unsupported(format!(
            "unsupported method {}/{}",
            config.method.id, config.method.version
        )));
    }
    Ok(())
}

pub fn secret_output_is_absent() -> Option<&'static str> {
    None
}

fn checked_vocab_len(vocab_size: u64) -> Result<usize> {
    if !(2..=u32::MAX as u64).contains(&vocab_size) {
        return Err(CompilerError::InvalidPlan {
            reason: "client secret vocabulary must be between 2 and u32::MAX".into(),
        });
    }
    usize::try_from(vocab_size).map_err(|_| CompilerError::ArithmeticOverflow {
        operation: "client secret vocabulary allocation",
    })
}

fn inverse_permutation(vocab_size: u64, permutation: &[u32]) -> Result<Vec<u32>> {
    let length = checked_vocab_len(vocab_size)?;
    if permutation.len() != length {
        return Err(CompilerError::Invariant(
            "client secret permutation length differs from vocabulary".into(),
        ));
    }
    let mut inverse = vec![u32::MAX; length];
    for (original, &obfuscated) in permutation.iter().enumerate() {
        let target = obfuscated as usize;
        if target >= length || inverse[target] != u32::MAX {
            return Err(CompilerError::Invariant(
                "client secret permutation is not a bijection".into(),
            ));
        }
        inverse[target] = original as u32;
    }
    Ok(inverse)
}

fn random_below(bound: u64) -> Result<u64> {
    debug_assert!(bound > 0);
    let range = u128::from(u64::MAX) + 1;
    let limit = range - range % u128::from(bound);
    loop {
        let mut bytes = [0_u8; 8];
        getrandom::fill(&mut bytes)
            .map_err(|error| CompilerError::Invariant(format!("OS randomness failed: {error}")))?;
        let value = u64::from_le_bytes(bytes);
        if u128::from(value) < limit {
            return Ok(value % bound);
        }
    }
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) -> Result<()> {
    let length = u32::try_from(value.len()).map_err(|_| CompilerError::ArithmeticOverflow {
        operation: "client secret commitment string length",
    })?;
    hasher.update(&length.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn decode_hex_32(value: &str, label: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(CompilerError::Invariant(format!(
            "client secret {label} must contain 64 hexadecimal characters"
        )));
    }
    if value != value.to_ascii_lowercase() {
        return Err(CompilerError::Invariant(format!(
            "client secret {label} must use lowercase hexadecimal"
        )));
    }
    let mut bytes = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks(2).enumerate() {
        let high = hex_digit(chunk[0]);
        let low = hex_digit(chunk[1]);
        match (high, low) {
            (Some(high), Some(low)) => bytes[index] = (high << 4) | low,
            _ => {
                return Err(CompilerError::Invariant(format!(
                    "client secret {label} is not hexadecimal"
                )));
            }
        }
    }
    Ok(bytes)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn validate_method_object(value: &Value) -> Result<()> {
    let method = value
        .get("method")
        .and_then(Value::as_object)
        .ok_or_else(|| CompilerError::Invariant("client secret method must be an object".into()))?;
    if method.len() != 2 || !method.contains_key("id") || !method.contains_key("version") {
        return Err(CompilerError::Invariant(
            "client secret method has unknown or missing fields".into(),
        ));
    }
    Ok(())
}

fn reject_duplicate_keys(bytes: &[u8]) -> std::result::Result<(), String> {
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

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    fn source() -> ModelFingerprint {
        ModelFingerprint::from_digest(blake3::hash(b"source"))
    }

    #[test]
    fn permutation_and_inverse_are_bijections() {
        let secret =
            ClientSecret::from_components(source(), 5, [7; 32], vec![2, 0, 4, 1, 3]).unwrap();
        secret.validate().unwrap();
        for (original, &obfuscated) in secret.permutation().iter().enumerate() {
            assert_eq!(
                secret.inverse_permutation()[obfuscated as usize],
                original as u32
            );
        }
    }

    #[test]
    fn commitment_changes_when_mapping_changes() {
        let first = ClientSecret::from_components(source(), 3, [7; 32], vec![1, 2, 0]).unwrap();
        let second = ClientSecret::from_components(source(), 3, [7; 32], vec![2, 0, 1]).unwrap();
        assert_ne!(first.secret_id, second.secret_id);
    }

    #[test]
    fn malformed_commitment_is_rejected() {
        let mut secret =
            ClientSecret::from_components(source(), 3, [7; 32], vec![1, 2, 0]).unwrap();
        secret.secret_id.replace_range(..2, "00");
        assert!(secret.validate().is_err());
    }

    #[test]
    fn duplicate_and_nested_unknown_json_keys_are_rejected() {
        assert!(reject_duplicate_keys(br#"{"version":1,"version":1}"#).is_err());
        let raw: Value = serde_json::from_str(
            r#"{"method":{"id":"aloepri-token","version":"0.1","extra":true}}"#,
        )
        .unwrap();
        assert!(validate_method_object(&raw).is_err());
    }
}
