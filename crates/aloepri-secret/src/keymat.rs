use crate::{
    decode_hex_32, hash_string, hex_encode, reject_duplicate_keys, validate_method_object,
};
use aloepri_core::{
    CompilerError, MethodContract, ModelFingerprint, Result,
    error::{io_error, json_error},
    keymat::{
        KEYMAT_ALGORITHM, KEYMAT_BALANCED_ALGORITHM, KEYMAT_RNG, KEYMAT_TOLERANCE, KeyMatBinding,
        lambda_bits, physical_hidden_size, validate_algorithm,
    },
};
use nalgebra::{DMatrix, linalg::SVD};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const FORMAT: &str = "keymat-f64-le-v1";
const DOMAIN: &[u8] = b"aloepri-keymat-secret-v1";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyMatSecretV1 {
    pub version: u32,
    pub method: MethodContract,
    pub secret_id: String,
    pub source_fingerprint: String,
    pub hidden_size: u64,
    pub expansion_size: u64,
    pub lambda_bits: u64,
    pub algorithm: String,
    pub rng: String,
    pub master_seed: String,
    pub nullspace_cutoff: f64,
    pub material_format: String,
    pub material_file: String,
    pub p_digest: String,
    pub q_digest: String,
}

pub struct KeyMaterial {
    d: usize,
    big_d: usize,
    p: Vec<f64>,
    q: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct KeyMatDiagnostics {
    pub finite_p: bool,
    pub finite_q: bool,
    pub max_abs_pq_error: f64,
    pub mean_abs_pq_error: f64,
    pub p_frobenius_norm: f64,
    pub q_frobenius_norm: f64,
    pub p_spectral_norm: f64,
    pub q_spectral_norm: f64,
    pub p_condition_estimate: f64,
}

impl KeyMaterial {
    pub fn hidden_size(&self) -> usize {
        self.d
    }
    pub fn physical_hidden_size(&self) -> usize {
        self.big_d
    }
    pub fn p(&self) -> &[f64] {
        &self.p
    }
    pub fn q(&self) -> &[f64] {
        &self.q
    }

    pub fn diagnostics(&self) -> Result<KeyMatDiagnostics> {
        self.validate()?;
        let (max_abs_pq_error, mean_abs_pq_error) = self.pq_errors();
        let p = DMatrix::from_row_slice(self.d, self.big_d, &self.p);
        let q = DMatrix::from_row_slice(self.big_d, self.d, &self.q);
        let ps = singular_values(p)?;
        let qs = singular_values(q)?;
        Ok(KeyMatDiagnostics {
            finite_p: true,
            finite_q: true,
            max_abs_pq_error,
            mean_abs_pq_error,
            p_frobenius_norm: self.p.iter().map(|v| v * v).sum::<f64>().sqrt(),
            q_frobenius_norm: self.q.iter().map(|v| v * v).sum::<f64>().sqrt(),
            p_spectral_norm: ps.iter().copied().fold(0.0, f64::max),
            q_spectral_norm: qs.iter().copied().fold(0.0, f64::max),
            p_condition_estimate: ps.iter().copied().fold(0.0, f64::max)
                / ps.iter().copied().fold(f64::INFINITY, f64::min),
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.d == 0
            || self.big_d <= self.d
            || self.p.len() != self.d * self.big_d
            || self.q.len() != self.p.len()
            || !self.p.iter().chain(&self.q).all(|v| v.is_finite())
        {
            return Err(invalid("invalid dimensions or non-finite KeyMat material"));
        }
        let (max, mean) = self.pq_errors();
        if !max.is_finite() || !mean.is_finite() || max > KEYMAT_TOLERANCE {
            return Err(invalid(&format!(
                "KeyMat PQ identity failed: max_abs={max:e}"
            )));
        }
        Ok(())
    }

    fn pq_errors(&self) -> (f64, f64) {
        let mut max = 0.0_f64;
        let mut sum = 0.0;
        for i in 0..self.d {
            for j in 0..self.d {
                let product: f64 = (0..self.big_d)
                    .map(|k| self.p[i * self.big_d + k] * self.q[k * self.d + j])
                    .sum();
                let error = (product - f64::from(i == j)).abs();
                if !error.is_finite() {
                    return (f64::INFINITY, f64::INFINITY);
                }
                max = max.max(error);
                sum += error;
            }
        }
        (max, sum / (self.d * self.d) as f64)
    }
}

impl KeyMatSecretV1 {
    pub fn generate(
        source: ModelFingerprint,
        d: u64,
        h: u64,
        lambda: f64,
        seed: Option<[u8; 32]>,
    ) -> Result<(Self, KeyMaterial, f64)> {
        Self::generate_with_algorithm(source, d, h, lambda, seed, KEYMAT_ALGORITHM)
    }

    pub fn generate_with_algorithm(
        source: ModelFingerprint,
        d: u64,
        h: u64,
        lambda: f64,
        seed: Option<[u8; 32]>,
        algorithm: &str,
    ) -> Result<(Self, KeyMaterial, f64)> {
        validate_algorithm(algorithm)?;
        let big_d = physical_hidden_size(d, h)?;
        aloepri_core::keymat::generation_peak_bytes(d, h)?;
        let lambda_bits = lambda_bits(lambda)?;
        let seed = match seed {
            Some(seed) => seed,
            None => {
                let mut seed = [0; 32];
                getrandom::fill(&mut seed)
                    .map_err(|e| invalid(&format!("OS randomness failed: {e}")))?;
                seed
            }
        };
        let (material, b_condition) = generate(
            d as usize,
            h as usize,
            f64::from_bits(lambda_bits),
            &seed,
            algorithm,
        )?;
        let mut secret = Self {
            version: 1,
            method: MethodContract::aloepri_keymat(),
            secret_id: String::new(),
            source_fingerprint: source.to_string(),
            hidden_size: d,
            expansion_size: h,
            lambda_bits,
            algorithm: algorithm.into(),
            rng: KEYMAT_RNG.into(),
            master_seed: hex_encode(&seed),
            nullspace_cutoff: 1e-10,
            material_format: FORMAT.into(),
            material_file: "key-material.bin".into(),
            p_digest: digest(material.p()),
            q_digest: digest(material.q()),
        };
        assert_eq!(material.physical_hidden_size(), big_d as usize);
        secret.secret_id = secret.compute_secret_id()?;
        secret.validate_material(&material)?;
        Ok((secret, material, b_condition))
    }

    pub fn binding(&self) -> Result<KeyMatBinding> {
        self.validate()?;
        Ok(KeyMatBinding {
            secret_id: self.secret_id.clone(),
            source_fingerprint: ModelFingerprint::from_hex(&self.source_fingerprint)?,
            hidden_size: self.hidden_size,
            expansion_size: self.expansion_size,
            physical_hidden_size: physical_hidden_size(self.hidden_size, self.expansion_size)?,
            lambda_bits: self.lambda_bits,
            algorithm: self.algorithm.clone(),
            rng: self.rng.clone(),
        })
    }

    pub fn validate(&self) -> Result<()> {
        validate_algorithm(&self.algorithm)?;
        physical_hidden_size(self.hidden_size, self.expansion_size)?;
        let source = ModelFingerprint::from_hex(&self.source_fingerprint)?;
        decode_hex_32(&self.master_seed, "KeyMat seed")?;
        decode_hex_32(&self.p_digest, "P digest")?;
        decode_hex_32(&self.q_digest, "Q digest")?;
        if self.version != 1
            || self.method != MethodContract::aloepri_keymat()
            || source.to_string() != self.source_fingerprint
            || self.rng != KEYMAT_RNG
            || lambda_bits(f64::from_bits(self.lambda_bits))? != self.lambda_bits
            || self.nullspace_cutoff != 1e-10
            || self.material_format != FORMAT
            || self.material_file != "key-material.bin"
            || self.secret_id != self.compute_secret_id()?
        {
            return Err(invalid(
                "KeyMat secret version, parameters or commitment mismatch",
            ));
        }
        Ok(())
    }

    pub fn validate_material(&self, material: &KeyMaterial) -> Result<()> {
        self.validate()?;
        if material.d as u64 != self.hidden_size
            || material.big_d as u64 != physical_hidden_size(self.hidden_size, self.expansion_size)?
            || digest(material.p()) != self.p_digest
            || digest(material.q()) != self.q_digest
        {
            return Err(invalid("KeyMat material identity mismatch"));
        }
        material.validate()
    }

    fn compute_secret_id(&self) -> Result<String> {
        let mut h = blake3::Hasher::new();
        h.update(DOMAIN);
        h.update(&self.version.to_le_bytes());
        for s in [
            &self.method.id,
            &self.method.version,
            &self.source_fingerprint,
            &self.algorithm,
            &self.rng,
            &self.master_seed,
            &self.material_format,
            &self.material_file,
            &self.p_digest,
            &self.q_digest,
        ] {
            hash_string(&mut h, s)?;
        }
        for n in [
            self.hidden_size,
            self.expansion_size,
            self.lambda_bits,
            self.nullspace_cutoff.to_bits(),
        ] {
            h.update(&n.to_le_bytes());
        }
        Ok(h.finalize().to_hex().to_string())
    }

    pub fn write_new(&self, path: &Path, material: &KeyMaterial) -> Result<()> {
        self.validate_material(material)?;
        reject_symlinks(path)?;
        let key_path = material_path(path);
        reject_symlinks(&key_path)?;
        if path.exists() || key_path.exists() {
            return Err(CompilerError::AlreadyExists {
                path: path.to_owned(),
            });
        }
        let mut keys = BufWriter::new(private_file(&key_path)?);
        for value in material.p.iter().chain(&material.q) {
            keys.write_all(&value.to_le_bytes())
                .map_err(|e| io_error(&key_path, e))?;
        }
        keys.flush().map_err(|e| io_error(&key_path, e))?;
        keys.get_ref()
            .sync_all()
            .map_err(|e| io_error(&key_path, e))?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| invalid(&e.to_string()))?;
        let mut json = private_file(path)?;
        json.write_all(&bytes).map_err(|e| io_error(path, e))?;
        json.sync_all().map_err(|e| io_error(path, e))?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| io_error(parent, e))?;
        Ok(())
    }

    pub fn read(path: &Path, memory_limit: u64) -> Result<(Self, KeyMaterial)> {
        reject_symlinks(path)?;
        let meta = fs::metadata(path).map_err(|e| io_error(path, e))?;
        if !meta.is_file() || meta.len() > 64 * 1024 {
            return Err(invalid("invalid KeyMat JSON file size"));
        }
        let bytes = fs::read(path).map_err(|e| io_error(path, e))?;
        reject_duplicate_keys(&bytes).map_err(|_| invalid("duplicate or invalid KeyMat JSON"))?;
        let raw = serde_json::from_slice(&bytes).map_err(|e| json_error(path, e))?;
        validate_method_object(&raw)?;
        let secret: Self = serde_json::from_slice(&bytes).map_err(|e| json_error(path, e))?;
        secret.validate()?;
        let state =
            aloepri_core::keymat::method_state_bytes(secret.hidden_size, secret.expansion_size)?;
        let validation_peak =
            aloepri_core::keymat::generation_peak_bytes(secret.hidden_size, secret.expansion_size)?;
        if validation_peak > memory_limit {
            return Err(CompilerError::MemoryLimitExceeded {
                requested: validation_peak,
                available: memory_limit,
            });
        }
        let key_path = material_path(path);
        reject_symlinks(&key_path)?;
        let metadata = fs::metadata(&key_path).map_err(|e| io_error(&key_path, e))?;
        if !metadata.is_file() || metadata.len() != state {
            return Err(invalid("invalid KeyMat binary length"));
        }
        let d = secret.hidden_size as usize;
        let big_d = physical_hidden_size(secret.hidden_size, secret.expansion_size)? as usize;
        let mut reader = BufReader::new(File::open(&key_path).map_err(|e| io_error(&key_path, e))?);
        let mut read_matrix = || -> Result<Vec<f64>> {
            let mut values = Vec::with_capacity(d * big_d);
            for _ in 0..d * big_d {
                let mut bytes = [0; 8];
                reader
                    .read_exact(&mut bytes)
                    .map_err(|e| io_error(&key_path, e))?;
                values.push(f64::from_le_bytes(bytes));
            }
            Ok(values)
        };
        let material = KeyMaterial {
            d,
            big_d,
            p: read_matrix()?,
            q: read_matrix()?,
        };
        secret.validate_material(&material)?;
        Ok((secret, material))
    }
}

pub fn material_path(path: &Path) -> PathBuf {
    path.with_file_name("key-material.bin")
}

fn private_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(|e| io_error(path, e))
}

pub fn reject_symlinks(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(invalid("KeyMat bundle paths must not contain symlinks"));
            }
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(io_error(ancestor, e));
            }
            _ => (),
        }
    }
    Ok(())
}

fn digest(values: &[f64]) -> String {
    let mut hash = blake3::Hasher::new();
    for v in values {
        hash.update(&v.to_le_bytes());
    }
    hash.finalize().to_hex().to_string()
}

fn invalid(reason: &str) -> CompilerError {
    CompilerError::InvalidPlan {
        reason: reason.into(),
    }
}

fn gaussian(rows: usize, cols: usize, seed: &[u8; 32], domain: &str, scale: f64) -> DMatrix<f64> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"aloepri-algorithm1-substream-v1");
    hash.update(seed);
    hash.update(domain.as_bytes());
    let mut rng = ChaCha20Rng::from_seed(*hash.finalize().as_bytes());
    DMatrix::from_fn(rows, cols, |_, _| {
        let v: f64 = StandardNormal.sample(&mut rng);
        v * scale
    })
}

fn orthogonal(n: usize, seed: &[u8; 32], domain: &str) -> DMatrix<f64> {
    let qr = gaussian(n, n, seed, domain, 1.0).qr();
    let mut q = qr.q();
    let r = qr.r();
    for col in 0..n {
        if r[(col, col)] < 0.0 {
            q.column_mut(col).scale_mut(-1.0);
        }
    }
    q
}

fn svd(matrix: DMatrix<f64>, vectors: bool) -> Result<SVD<f64, nalgebra::Dyn, nalgebra::Dyn>> {
    SVD::try_new(matrix, false, vectors, 1e-14, 100_000)
        .ok_or_else(|| invalid("KeyMat SVD did not converge"))
}

fn singular_values(matrix: DMatrix<f64>) -> Result<Vec<f64>> {
    Ok(svd(matrix, false)?
        .singular_values
        .iter()
        .copied()
        .collect())
}

fn nullspace(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let rows = matrix.nrows().max(matrix.ncols());
    let mut padded = DMatrix::zeros(rows, matrix.ncols());
    padded.view_mut((0, 0), matrix.shape()).copy_from(matrix);
    let svd = svd(padded, true)?;
    let cutoff = 1e-10_f64.max(1e-10 * svd.singular_values.iter().copied().fold(0.0, f64::max));
    let vt = svd
        .v_t
        .ok_or_else(|| invalid("KeyMat SVD omitted right vectors"))?;
    let indexes: Vec<_> = svd
        .singular_values
        .iter()
        .enumerate()
        .filter_map(|(i, s)| (*s <= cutoff).then_some(i))
        .collect();
    if indexes.is_empty() {
        return Err(invalid("KeyMat nullspace is empty"));
    }
    Ok(DMatrix::from_fn(matrix.ncols(), indexes.len(), |i, j| {
        vt[(indexes[j], i)]
    }))
}

fn generate(
    d: usize,
    h: usize,
    lambda: f64,
    seed: &[u8; 32],
    algorithm: &str,
) -> Result<(KeyMaterial, f64)> {
    let scale = 1.0 / (d as f64).sqrt();
    let null_scale = match algorithm {
        KEYMAT_ALGORITHM => 1.0,
        KEYMAT_BALANCED_ALGORITHM => scale,
        _ => return Err(invalid("unsupported KeyMat generation algorithm")),
    };
    let b = orthogonal(d, seed, "U") + gaussian(d, d, seed, "V", scale) * lambda;
    let singular = singular_values(b.clone())?;
    let b_condition = singular.iter().copied().fold(0.0, f64::max)
        / singular.iter().copied().fold(f64::INFINITY, f64::min);
    if !b_condition.is_finite() {
        return Err(invalid("KeyMat B is singular"));
    }
    let b_inv = b
        .clone()
        .try_inverse()
        .ok_or_else(|| invalid("KeyMat B is not invertible"))?;
    let e = gaussian(d, h / 2, seed, "E1", scale) * gaussian(h / 2, h, seed, "E2", scale);
    let f = gaussian(h, h / 2, seed, "F1", scale) * gaussian(h / 2, d, seed, "F2", scale);
    let fb = nullspace(&f.transpose())?;
    let c = gaussian(d, fb.ncols(), seed, "C", null_scale) * fb.transpose();
    let eb = nullspace(&e)?;
    let n = &eb * gaussian(eb.ncols(), d, seed, "N", null_scale);
    let big_d = d + 2 * h;
    let z = orthogonal(big_d, seed, "Z");
    let mut left = DMatrix::zeros(d, big_d);
    left.view_mut((0, 0), (d, d)).copy_from(&b);
    left.view_mut((0, d), (d, h)).copy_from(&c);
    left.view_mut((0, d + h), (d, h)).copy_from(&e);
    let mut right = DMatrix::zeros(big_d, d);
    right.view_mut((0, 0), (d, d)).copy_from(&b_inv);
    right.view_mut((d, 0), (h, d)).copy_from(&f);
    right.view_mut((d + h, 0), (h, d)).copy_from(&n);
    let p = left * z.clone();
    let q = z.transpose() * right;
    let material = KeyMaterial {
        d,
        big_d,
        p: row_major(&p),
        q: row_major(&q),
    };
    material.validate()?;
    Ok((material, b_condition))
}

fn row_major(matrix: &DMatrix<f64>) -> Vec<f64> {
    let mut values = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            values.push(matrix[(row, column)]);
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> ModelFingerprint {
        ModelFingerprint::from_digest(blake3::hash(b"source"))
    }
    #[test]
    fn algorithm1_identity_and_repeatability() {
        for (d, h, lambda) in [(8, 2, 0.3), (4, 8, 0.0), (16, 4, 0.1)] {
            let (a, pq, condition) =
                KeyMatSecretV1::generate(source(), d, h, lambda, Some([42; 32])).unwrap();
            let (b, other, _) =
                KeyMatSecretV1::generate(source(), d, h, lambda, Some([42; 32])).unwrap();
            assert_eq!(a.secret_id, b.secret_id);
            assert_eq!(pq.p, other.p);
            assert_eq!(pq.q, other.q);
            assert_eq!(pq.p.capacity(), pq.p.len());
            assert_eq!(pq.q.capacity(), pq.q.len());
            assert_eq!(pq.d, d as usize);
            assert_eq!(pq.big_d, (d + 2 * h) as usize);
            assert!(pq.diagnostics().unwrap().max_abs_pq_error < 1e-10);
            assert!(condition.is_finite());
            assert!(pq.p.iter().any(|v| v.abs() > 0.1));
        }
    }
    #[test]
    fn v1_golden_and_versioned_identity_remain_stable() {
        let source = ModelFingerprint::from_hex(
            "5a22a0ce4810146973bf43b941a7a530fab6147abd4354c6d0ab0cf5666af431",
        )
        .unwrap();
        let (v1, original, _) =
            KeyMatSecretV1::generate(source, 8, 2, 0.3, Some([42; 32])).unwrap();
        assert_eq!(
            v1.p_digest,
            "0cd0e4344a8f2ff6cf28c040df4e1e2fa069a49941c698073686153cd9120c40"
        );
        assert_eq!(
            v1.q_digest,
            "725f87f6a1355241e593901b0e8fba75f1094ce27e73a167fc641964a666fe2e"
        );
        assert_eq!(
            v1.secret_id,
            "bb09de48c02e788add2b65c992c2117ea778534602572dc92a6c345345fd6b83"
        );
        let (explicit, same, _) = KeyMatSecretV1::generate_with_algorithm(
            source,
            8,
            2,
            0.3,
            Some([42; 32]),
            KEYMAT_ALGORITHM,
        )
        .unwrap();
        assert_eq!(explicit.secret_id, v1.secret_id);
        assert_eq!(same.p, original.p);
        assert_eq!(same.q, original.q);
        let (v2, balanced, _) = KeyMatSecretV1::generate_with_algorithm(
            source,
            8,
            2,
            0.3,
            Some([42; 32]),
            KEYMAT_BALANCED_ALGORITHM,
        )
        .unwrap();
        let (again, repeated, _) = KeyMatSecretV1::generate_with_algorithm(
            source,
            8,
            2,
            0.3,
            Some([42; 32]),
            KEYMAT_BALANCED_ALGORITHM,
        )
        .unwrap();
        assert_eq!(v2.secret_id, again.secret_id);
        assert_eq!(balanced.p, repeated.p);
        assert_eq!(balanced.q, repeated.q);
        assert_ne!(v1.secret_id, v2.secret_id);
        assert_ne!(v1.p_digest, v2.p_digest);
        assert_ne!(v1.q_digest, v2.q_digest);
        let mut tampered = v1.clone();
        tampered.algorithm = KEYMAT_BALANCED_ALGORITHM.into();
        assert!(tampered.validate().is_err());
        for algorithm in ["", "algorithm1-v1-extra", "algorithm1-balanced-null-v3"] {
            assert!(
                KeyMatSecretV1::generate_with_algorithm(
                    source,
                    8,
                    2,
                    0.3,
                    Some([42; 32]),
                    algorithm
                )
                .is_err()
            );
        }
    }

    #[test]
    fn balanced_nullspace_preserves_bases_cross_terms_and_gram_relations() {
        let seed = [7; 32];
        for (d, h) in [(8, 2), (4, 8), (1, 2)] {
            let (v1, original, _) =
                KeyMatSecretV1::generate(source(), d, h, 0.3, Some(seed)).unwrap();
            let (v2, balanced, _) = KeyMatSecretV1::generate_with_algorithm(
                source(),
                d,
                h,
                0.3,
                Some(seed),
                KEYMAT_BALANCED_ALGORITHM,
            )
            .unwrap();
            let d = d as usize;
            let h = h as usize;
            let big_d = d + 2 * h;
            let scale = 1.0 / (d as f64).sqrt();
            let z = orthogonal(big_d, &seed, "Z");
            let p1 = DMatrix::from_row_slice(d, big_d, &original.p);
            let q1 = DMatrix::from_row_slice(big_d, d, &original.q);
            let p2 = DMatrix::from_row_slice(d, big_d, &balanced.p);
            let q2 = DMatrix::from_row_slice(big_d, d, &balanced.q);
            let left1 = &p1 * z.transpose();
            let left2 = &p2 * z.transpose();
            let right1 = &z * &q1;
            let right2 = &z * &q2;
            let b = left1.columns(0, d).into_owned();
            let c = left1.columns(d, h).into_owned();
            let e = left1.columns(d + h, h).into_owned();
            let inverse = right1.rows(0, d).into_owned();
            let f = right1.rows(d, h).into_owned();
            let n = right1.rows(d + h, h).into_owned();
            assert!((left2.columns(0, d) - &b).amax() < 1e-11);
            assert!((left2.columns(d, h) - &c * scale).amax() < 1e-11);
            assert!((left2.columns(d + h, h) - &e).amax() < 1e-11);
            assert!((right2.rows(0, d) - &inverse).amax() < 1e-11);
            assert!((right2.rows(d, h) - &f).amax() < 1e-11);
            assert!((right2.rows(d + h, h) - &n * scale).amax() < 1e-11);
            assert!((&c * &f).amax() < 1e-10);
            assert!((&e * &n).amax() < 1e-10);
            let p_expected =
                &b * b.transpose() + (&c * c.transpose()) * (scale * scale) + &e * e.transpose();
            let q_expected = inverse.transpose() * &inverse
                + f.transpose() * &f
                + (n.transpose() * &n) * (scale * scale);
            assert!((&p2 * p2.transpose() - p_expected).amax() < 1e-10);
            assert!((q2.transpose() * &q2 - q_expected).amax() < 1e-10);
            assert!((&p2 * &q2 - DMatrix::identity(d, d)).amax() < 1e-10);
            let p1s = singular_values(p1).unwrap();
            let q1s = singular_values(q1).unwrap();
            let p2s = singular_values(p2).unwrap();
            let q2s = singular_values(q2).unwrap();
            let bs = singular_values(b).unwrap();
            let inverse_s = singular_values(inverse).unwrap();
            assert!(
                p2s.iter().copied().fold(0.0, f64::max)
                    <= p1s.iter().copied().fold(0.0, f64::max) + 1e-10
            );
            assert!(
                q2s.iter().copied().fold(0.0, f64::max)
                    <= q1s.iter().copied().fold(0.0, f64::max) + 1e-10
            );
            assert!(
                p2s.iter().copied().fold(f64::INFINITY, f64::min)
                    >= bs.iter().copied().fold(f64::INFINITY, f64::min) - 1e-10
            );
            assert!(
                q2s.iter().copied().fold(f64::INFINITY, f64::min)
                    >= inverse_s.iter().copied().fold(f64::INFINITY, f64::min) - 1e-10
            );
            assert_ne!(v1.secret_id, v2.secret_id);
            if d == 1 {
                assert_eq!(original.p, balanced.p);
                assert_eq!(original.q, balanced.q);
            } else {
                assert_ne!(original.p, balanced.p);
                assert_ne!(original.q, balanced.q);
            }
            assert!(balanced.p.iter().any(|v| v.abs() > 0.1));
        }
    }

    #[test]
    fn independent_nullspace_and_qr_checks() {
        let seed = [7; 32];
        let z = orthogonal(12, &seed, "Z");
        assert!((&z * z.transpose() - DMatrix::identity(12, 12)).amax() < 1e-12);
        let e = gaussian(4, 3, &seed, "E", 1.0) * gaussian(3, 8, &seed, "F", 1.0);
        let n = nullspace(&e).unwrap();
        assert!((&e * &n).amax() < 1e-10);
        let f = e.transpose();
        let c = nullspace(&f.transpose()).unwrap().transpose();
        assert!((c * f).amax() < 1e-10);
    }
    #[test]
    fn invalid_inputs_and_tampering_fail_closed() {
        for (d, h, lambda) in [
            (0, 2, 0.3),
            (2, 0, 0.3),
            (2, 3, 0.3),
            (2, 2, -1.0),
            (2, 2, f64::NAN),
            (2, 2, f64::INFINITY),
        ] {
            assert!(KeyMatSecretV1::generate(source(), d, h, lambda, Some([1; 32])).is_err());
        }
        let (mut secret, mut keys, _) =
            KeyMatSecretV1::generate(source(), 4, 2, 0.3, Some([1; 32])).unwrap();
        secret.master_seed = hex_encode(&[2; 32]);
        assert!(secret.validate().is_err());
        keys.p[0] = f64::NAN;
        assert!(keys.validate().is_err());
    }
    #[test]
    fn bundle_roundtrip_and_corruption() {
        let root = std::env::temp_dir().join(format!("keymat-secret-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("secret.json");
        let (secret, keys, _) =
            KeyMatSecretV1::generate(source(), 4, 2, 0.3, Some([1; 32])).unwrap();
        secret.write_new(&path, &keys).unwrap();
        let (loaded, read) = KeyMatSecretV1::read(&path, u64::MAX).unwrap();
        assert_eq!(loaded.secret_id, secret.secret_id);
        assert_eq!(read.p, keys.p);
        assert_eq!(read.q, keys.q);
        assert!(secret.write_new(&path, &keys).is_err());
        fs::write(material_path(&path), [0; 8]).unwrap();
        assert!(KeyMatSecretV1::read(&path, u64::MAX).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
