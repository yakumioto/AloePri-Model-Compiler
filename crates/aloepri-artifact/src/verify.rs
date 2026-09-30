use crate::{
    fingerprint::tensor_fingerprint,
    hf::HfArtifact,
    layout::validate_output_layout,
    manifest::{MANIFEST_VERSION, Manifest, TOKEN_MANIFEST_VERSION, TensorManifest},
};
use aloepri_core::{
    backend::VerificationOutcome,
    error::{CompilerError, Result},
    model::ModelArtifact,
    plan::{OperationKind, SCHEMA_VERSION, TransformPlan},
    types::TensorName,
};
use std::{collections::BTreeMap, path::Path};

/// Structural verification of a published artifact.
///
/// This is deliberately *not* model semantic verification: it proves the
/// container and, for a v3 contract-complete artifact, that the manifest,
/// plan, layout and physical shards agree. It never loads a model and never
/// claims a non-standard checkpoint is loadable by vanilla Transformers.
pub fn verify_artifact(root: &Path) -> Result<VerificationOutcome> {
    let artifact = HfArtifact::open(root)?;
    let fingerprint = artifact.fingerprint()?;
    let manifest_path = root.join("aloepri.json");
    let Some(manifest) = (if manifest_path.is_file() {
        Some(Manifest::read(root)?)
    } else {
        None
    }) else {
        return Ok(VerificationOutcome {
            structure_valid: true,
            manifest_verified: false,
            artifact_valid: false,
            plan_verified: false,
            standard_hf_checkpoint: None,
            runtime_required: false,
            verification_scope: "container".into(),
            semantic_verification: "not_run".into(),
            runtime_contract: None,
            tensor_count: artifact.tensors().len(),
            payload_bytes: artifact.total_payload_bytes(),
            artifact_fingerprint: fingerprint.to_string(),
        });
    };
    match manifest.artifact_version {
        SCHEMA_VERSION | aloepri_core::keymat::KEYMAT_SCHEMA_VERSION => {
            verify_current(&artifact, &manifest, fingerprint)
        }
        MANIFEST_VERSION | TOKEN_MANIFEST_VERSION => {
            verify_legacy(&artifact, &manifest, fingerprint)
        }
        version => Err(CompilerError::UnsupportedVersion { version }),
    }
}

fn verify_legacy(
    artifact: &HfArtifact,
    manifest: &Manifest,
    fingerprint: aloepri_core::ModelFingerprint,
) -> Result<VerificationOutcome> {
    check_common(artifact, manifest)?;
    let expected = tensor_index(&manifest.tensors)?;
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
        check_tensor(artifact, descriptor, item)?;
    }
    Ok(VerificationOutcome {
        structure_valid: true,
        manifest_verified: true,
        artifact_valid: true,
        plan_verified: false,
        standard_hf_checkpoint: Some(true),
        runtime_required: false,
        verification_scope: "legacy_v1_v2".into(),
        semantic_verification: "not_run".into(),
        runtime_contract: None,
        tensor_count: artifact.tensors().len(),
        payload_bytes: artifact.total_payload_bytes(),
        artifact_fingerprint: fingerprint.to_string(),
    })
}

fn verify_current(
    artifact: &HfArtifact,
    manifest: &Manifest,
    fingerprint: aloepri_core::ModelFingerprint,
) -> Result<VerificationOutcome> {
    check_common(artifact, manifest)?;
    let plan = manifest
        .plan
        .as_ref()
        .ok_or_else(|| CompilerError::OutputCorrupted {
            reason: "v3 manifest is missing its plan".into(),
        })?;
    let runtime =
        manifest
            .runtime_contract
            .as_ref()
            .ok_or_else(|| CompilerError::OutputCorrupted {
                reason: "v3 manifest is missing its runtime contract".into(),
            })?;
    check_logical_dimensions(artifact, plan)?;
    check_physical_dimensions(plan)?;
    validate_output_layout(&manifest.output_layout)?;

    let expected = tensor_index(&manifest.tensors)?;
    if expected.len() != manifest.tensors.len()
        || expected.len() != artifact.tensors().len()
        || expected.len() != plan.output_inventory.len()
    {
        return Err(CompilerError::OutputCorrupted {
            reason: "manifest, plan and artifact tensor inventories differ".into(),
        });
    }
    for descriptor in artifact.tensors() {
        let item =
            expected
                .get(&descriptor.name)
                .ok_or_else(|| CompilerError::OutputCorrupted {
                    reason: format!("manifest is missing tensor {}", descriptor.name),
                })?;
        check_tensor(artifact, descriptor, item)?;
    }
    check_physical_layout(artifact, plan)?;
    Ok(VerificationOutcome {
        structure_valid: true,
        manifest_verified: true,
        artifact_valid: true,
        plan_verified: true,
        standard_hf_checkpoint: Some(runtime.standard_hf_checkpoint),
        runtime_required: !runtime.standard_hf_checkpoint,
        verification_scope: format!("v{}_contract", plan.version),
        semantic_verification: "not_run".into(),
        runtime_contract: Some(runtime.clone()),
        tensor_count: artifact.tensors().len(),
        payload_bytes: artifact.total_payload_bytes(),
        artifact_fingerprint: fingerprint.to_string(),
    })
}

fn check_common(artifact: &HfArtifact, manifest: &Manifest) -> Result<()> {
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
    Ok(())
}

fn check_logical_dimensions(artifact: &HfArtifact, plan: &TransformPlan) -> Result<()> {
    let config = artifact.config();
    for (key, value) in &plan.runtime_contract.logical_dimensions {
        if config.get(key).and_then(serde_json::Value::as_u64) != Some(*value) {
            return Err(CompilerError::OutputCorrupted {
                reason: format!("runtime logical dimension `{key}` disagrees with the config"),
            });
        }
    }
    Ok(())
}

fn check_physical_dimensions(plan: &TransformPlan) -> Result<()> {
    let runtime = &plan.runtime_contract;
    for operation in &plan.operations {
        if let OperationKind::PadColumns { .. } = operation.kind {
            let physical_hidden = runtime.physical_dimensions.get("hidden_size").copied();
            let output_hidden = operation.output.descriptor.shape.as_slice().get(1).copied();
            if physical_hidden.is_none() || physical_hidden != output_hidden {
                return Err(CompilerError::OutputCorrupted {
                    reason: "padded output does not match the physical hidden size".into(),
                });
            }
        }
    }
    Ok(())
}

fn check_physical_layout(artifact: &HfArtifact, plan: &TransformPlan) -> Result<()> {
    let actual_shards = artifact.shards();
    let expected_shards = &plan.output_layout.shards;
    if actual_shards.len() != expected_shards.len() {
        return Err(CompilerError::OutputCorrupted {
            reason: "the artifact shard count differs from the planned layout".into(),
        });
    }
    let mut filename_to_index = BTreeMap::new();
    for (index, shard) in actual_shards.iter().enumerate() {
        filename_to_index.insert(shard.filename.clone(), index);
    }
    for planned in expected_shards {
        let index = filename_to_index.get(&planned.filename).ok_or_else(|| {
            CompilerError::OutputCorrupted {
                reason: format!("planned shard {} is missing", planned.filename),
            }
        })?;
        let actual = &actual_shards[*index];
        if actual.file_length != planned.file_length.0
            || actual.payload_length != planned.payload_length.0
        {
            return Err(CompilerError::OutputCorrupted {
                reason: format!("shard {} does not match its planned size", planned.filename),
            });
        }
    }
    for tensor in &plan.output_layout.tensors {
        let descriptor = artifact
            .tensors()
            .iter()
            .find(|descriptor| descriptor.name == tensor.name)
            .ok_or_else(|| CompilerError::OutputCorrupted {
                reason: format!("physical tensor {} is missing", tensor.name),
            })?;
        let shard = &actual_shards[descriptor.location.shard.0 as usize];
        let planned = &expected_shards[tensor.shard as usize];
        if shard.filename != planned.filename {
            return Err(CompilerError::OutputCorrupted {
                reason: format!("tensor {} lives in the wrong shard", tensor.name),
            });
        }
        let data_start = shard.file_length - shard.payload_length;
        let relative = descriptor.location.offset.0 - data_start;
        if relative != tensor.offset.0 {
            return Err(CompilerError::OutputCorrupted {
                reason: format!("tensor {} has an unexpected relative offset", tensor.name),
            });
        }
    }
    Ok(())
}

fn tensor_index(tensors: &[TensorManifest]) -> Result<BTreeMap<TensorName, &TensorManifest>> {
    let mut map = BTreeMap::new();
    for tensor in tensors {
        if map.insert(tensor.name.clone(), tensor).is_some() {
            return Err(CompilerError::OutputCorrupted {
                reason: "manifest lists a tensor twice".into(),
            });
        }
    }
    Ok(map)
}

fn check_tensor(
    artifact: &HfArtifact,
    descriptor: &aloepri_core::types::TensorDescriptor,
    expected: &TensorManifest,
) -> Result<()> {
    if expected.shape != descriptor.shape
        || expected.dtype != descriptor.dtype
        || expected.byte_length != descriptor.byte_length.0
    {
        return Err(CompilerError::OutputCorrupted {
            reason: format!("manifest metadata differs for {}", descriptor.name),
        });
    }
    let mut reader = artifact.tensor_reader(&descriptor.name)?;
    let digest = tensor_fingerprint(reader.as_mut(), descriptor)?.to_string();
    if digest != expected.blake3 {
        return Err(CompilerError::OutputCorrupted {
            reason: format!("tensor digest differs for {}", descriptor.name),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aloepri_core::{
        plan::{MethodContract, OutputLayout},
        types::{DType, TensorName, TensorShape},
    };
    use std::{fs, io::Write};
    use tempfile::tempdir;

    const PAYLOAD: [u8; 3] = [1, 2, 3];

    fn write_container(root: &Path) {
        fs::write(root.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        let header = br#"{"x":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#;
        let mut file = fs::File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&PAYLOAD).unwrap();
    }

    fn legacy_manifest(version: u32, method: MethodContract) -> Manifest {
        Manifest {
            artifact_version: version,
            method: method.clone(),
            architecture: "llama".into(),
            source_fingerprint: "11".repeat(32),
            plan_hash: "22".repeat(32),
            layout_hash: None,
            runtime_contract: None,
            plan: None,
            output_layout: OutputLayout {
                shards: vec![],
                tensors: vec![],
                index: None,
            },
            config_blake3: blake3::hash(br#"{"model_type":"llama"}"#)
                .to_hex()
                .to_string(),
            sidecar_blake3: BTreeMap::new(),
            tensors: vec![TensorManifest {
                name: TensorName::try_from("x").unwrap(),
                shape: TensorShape::new(vec![3]),
                dtype: DType::U8,
                byte_length: 3,
                blake3: blake3::hash(&PAYLOAD).to_hex().to_string(),
            }],
            standard_hf_checkpoint: true,
            secret_key_id: None,
            secret_id: (method != MethodContract::identity()).then(|| "33".repeat(32)),
        }
    }

    #[test]
    fn legacy_identity_and_token_manifests_stay_read_only_verifiable() {
        for (version, method) in [
            (MANIFEST_VERSION, MethodContract::identity()),
            (TOKEN_MANIFEST_VERSION, MethodContract::aloepri_token()),
        ] {
            let directory = tempdir().unwrap();
            write_container(directory.path());
            legacy_manifest(version, method)
                .write(directory.path())
                .unwrap();
            let outcome = verify_artifact(directory.path()).unwrap();
            assert!(outcome.manifest_verified);
            assert!(outcome.artifact_valid);
            // A legacy artifact carries no v3 plan integrity proof.
            assert!(!outcome.plan_verified);
            assert_eq!(outcome.verification_scope, "legacy_v1_v2");
            assert_eq!(outcome.standard_hf_checkpoint, Some(true));
        }
    }

    #[test]
    fn legacy_manifest_with_v3_fields_is_rejected() {
        let directory = tempdir().unwrap();
        write_container(directory.path());
        let mut manifest = legacy_manifest(MANIFEST_VERSION, MethodContract::identity());
        manifest.layout_hash = Some("44".repeat(32));
        manifest.write(directory.path()).unwrap();
        assert!(verify_artifact(directory.path()).is_err());
    }

    #[test]
    fn unmanifested_container_is_structure_only() {
        let directory = tempdir().unwrap();
        write_container(directory.path());
        let outcome = verify_artifact(directory.path()).unwrap();
        assert!(outcome.structure_valid);
        assert!(!outcome.manifest_verified);
        assert!(!outcome.artifact_valid);
        assert_eq!(outcome.standard_hf_checkpoint, None);
        assert_eq!(outcome.verification_scope, "container");
    }
}
