use crate::{
    fingerprint::tensor_fingerprint,
    hf::HfArtifact,
    manifest::{Manifest, TensorManifest},
};
use aloepri_core::{
    backend::VerificationOutcome,
    error::{CompilerError, Result},
    model::ModelArtifact,
    types::TensorName,
};
use std::{collections::BTreeMap, path::Path};

pub fn verify_artifact(root: &Path) -> Result<VerificationOutcome> {
    let artifact = HfArtifact::open(root)?;
    let fingerprint = artifact.fingerprint()?;
    let manifest_path = root.join("aloepri.json");
    let manifest = if manifest_path.is_file() {
        Some(Manifest::read(root)?)
    } else {
        None
    };
    let mut manifest_verified = false;
    if let Some(manifest) = manifest {
        if !manifest.standard_hf_checkpoint || manifest.method.id != "identity" {
            return Err(CompilerError::OutputCorrupted {
                reason: "manifest is not an identity standard HF checkpoint".into(),
            });
        }
        let config_hash = blake3::hash(artifact.config_bytes()).to_hex().to_string();
        if config_hash != manifest.config_blake3 {
            return Err(CompilerError::OutputCorrupted {
                reason: "config digest does not match manifest".into(),
            });
        }
        let sidecars = artifact.sidecar_bytes()?;
        let sidecar_hashes: BTreeMap<_, _> = sidecars
            .iter()
            .map(|(name, bytes)| (name.clone(), blake3::hash(bytes).to_hex().to_string()))
            .collect();
        if sidecar_hashes != manifest.sidecar_blake3 {
            return Err(CompilerError::OutputCorrupted {
                reason: "sidecar digests do not match manifest".into(),
            });
        }
        let expected: BTreeMap<TensorName, &TensorManifest> = manifest
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect();
        if expected.len() != manifest.tensors.len() || expected.len() != artifact.tensors().len() {
            return Err(CompilerError::OutputCorrupted {
                reason: "manifest tensor inventory differs from artifact".into(),
            });
        }
        for descriptor in artifact.tensors() {
            let item =
                expected
                    .get(&descriptor.name)
                    .ok_or_else(|| CompilerError::OutputCorrupted {
                        reason: format!("manifest is missing tensor {}", descriptor.name),
                    })?;
            if item.shape != descriptor.shape
                || item.dtype != descriptor.dtype
                || item.byte_length != descriptor.byte_length.0
            {
                return Err(CompilerError::OutputCorrupted {
                    reason: format!("manifest metadata differs for {}", descriptor.name),
                });
            }
            let mut reader = artifact.tensor_reader(&descriptor.name)?;
            let digest = tensor_fingerprint(reader.as_mut(), descriptor)?.to_string();
            if digest != item.blake3 {
                return Err(CompilerError::OutputCorrupted {
                    reason: format!("tensor digest differs for {}", descriptor.name),
                });
            }
        }
        manifest_verified = true;
    }
    Ok(VerificationOutcome {
        structure_valid: true,
        manifest_verified,
        tensor_count: artifact.tensors().len(),
        payload_bytes: artifact.total_payload_bytes(),
        artifact_fingerprint: fingerprint.to_string(),
    })
}
