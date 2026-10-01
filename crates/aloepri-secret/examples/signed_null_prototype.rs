use aloepri_core::{
    ModelFingerprint,
    keymat::{generation_peak_bytes, lambda_bits, physical_hidden_size},
};
use aloepri_secret::keymat::KeyMatSecretV1;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

const ALGORITHM: &str = "algorithm1-signed-null-v3";

fn rng(seed: &[u8; 32], domain: &str) -> ChaCha20Rng {
    let mut h = blake3::Hasher::new();
    h.update(b"aloepri-algorithm1-signed-null-v3-substream");
    h.update(seed);
    h.update(domain.as_bytes());
    ChaCha20Rng::from_seed(*h.finalize().as_bytes())
}
fn gaussian(rows: usize, cols: usize, seed: &[u8; 32], domain: &str, scale: f64) -> Vec<f64> {
    let mut rng = rng(seed, domain);
    (0..rows * cols)
        .map(|_| {
            let value: f64 = StandardNormal.sample(&mut rng);
            value * scale
        })
        .collect()
}
fn material(d: usize, h: usize, lambda: f64, seed: &[u8; 32]) -> (Vec<f64>, Vec<f64>) {
    let r = h / 2;
    let big_d = d + 2 * h;
    let scale = (1.0 + lambda) / (d as f64).sqrt();
    let mut signs = rng(seed, "S");
    let mut core: Vec<f64> = (0..d)
        .map(|_| if signs.next_u32() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    core[0] = -1.0;
    let mut aux = rng(seed, "T");
    let mut permutation: Vec<usize> = (0..2 * h).collect();
    for index in (1..2 * h).rev() {
        let swap = aux.random_range(0..=index);
        permutation.swap(index, swap);
    }
    let aux_signs: Vec<f64> = (0..2 * h)
        .map(|_| if aux.next_u32() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let c = gaussian(d, r, seed, "Cg", scale);
    let e = gaussian(d, r, seed, "Eg", scale);
    let f = gaussian(r, d, seed, "Fg", scale);
    let n = gaussian(r, d, seed, "Ng", scale);
    let mut p = vec![0.0; d * big_d];
    let mut q = vec![0.0; big_d * d];
    for i in 0..d {
        p[i * big_d + i] = core[i];
        q[i * d + i] = core[i];
        for j in 0..r {
            p[i * big_d + d + permutation[r + j]] = c[i * r + j] * aux_signs[r + j];
            p[i * big_d + d + permutation[h + j]] = e[i * r + j] * aux_signs[h + j];
            q[(d + permutation[j]) * d + i] = f[j * d + i] * aux_signs[j];
            q[(d + permutation[h + r + j]) * d + i] = n[j * d + i] * aux_signs[h + r + j];
        }
    }
    (p, q)
}
fn digest(values: &[f64]) -> String {
    let mut h = blake3::Hasher::new();
    for value in values {
        h.update(&value.to_le_bytes());
    }
    h.finalize().to_hex().to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: signed_null_prototype CONTROL_SECRET_JSON PRIVATE_OUTPUT_DIR".into());
    }
    let control: KeyMatSecretV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    control.validate()?;
    let _ = ModelFingerprint::from_hex(&control.source_fingerprint)?;
    let big_d = physical_hidden_size(control.hidden_size, control.expansion_size)?;
    let peak = generation_peak_bytes(control.hidden_size, control.expansion_size)?;
    if peak > 256 * 1024 * 1024 {
        return Err("prototype exceeds fixed256MiB generation budget".into());
    }
    let d = usize::try_from(control.hidden_size)?;
    let h = usize::try_from(control.expansion_size)?;
    let lambda = f64::from_bits(control.lambda_bits);
    lambda_bits(lambda)?;
    let mut seed = [0; 32];
    for (i, bytes) in control
        .master_seed
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
    {
        seed[i] = u8::from_str_radix(std::str::from_utf8(bytes)?, 16)?;
    }
    let (p, q) = material(d, h, lambda, &seed);
    if !p.iter().chain(&q).all(|v| v.is_finite()) {
        return Err("non-finite prototype keys".into());
    }
    let mut max = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            let dot: f64 = (0..big_d as usize)
                .map(|k| p[i * big_d as usize + k] * q[k * d + j])
                .sum();
            max = max.max((dot - f64::from(i == j)).abs());
        }
    }
    if max > 1e-5 {
        return Err("prototype G1 failed".into());
    }
    let output = Path::new(&args[2]);
    let path = output.join("material.bin");
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = BufWriter::new(options.open(&path)?);
    for value in p.iter().chain(&q) {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    file.get_ref().sync_all()?;
    File::open(output)?.sync_all()?;
    let metadata = serde_json::json!({"diagnostic_only":true,"production_algorithm_added":false,"algorithm":ALGORITHM,"d":d,"h":h,"D":big_d,"lambda_bits":control.lambda_bits,"rng":control.rng,"source_fingerprint":control.source_fingerprint,"control_secret_id":control.secret_id,"p_digest":digest(&p),"q_digest":digest(&q),"finite_p":true,"finite_q":true,"max_abs_pq_error":max,"generation_peak_bytes":peak,"coefficient_std":(1.0+lambda)/(d as f64).sqrt(),"B_perturbation_V_zero":true});
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disjoint_support_and_signed_core_are_not_padding() {
        for (d, h, lambda) in [(1, 2, 0.0), (4, 8, 0.3), (8, 2, 0.3)] {
            let (p, q) = material(d, h, lambda, &[7; 32]);
            let big_d = d + 2 * h;
            assert_eq!(p[0], -1.0);
            assert!(p.iter().chain(&q).all(|v| v.is_finite()));
            assert!((0..d).any(|i| (d..big_d).any(|j| p[i * big_d + j] != 0.0)));
            for column in d..big_d {
                let p_active = (0..d).any(|i| p[i * big_d + column] != 0.0);
                let q_active = (0..d).any(|i| q[column * d + i] != 0.0);
                assert_ne!(p_active, q_active);
            }
            for i in 0..d {
                for j in 0..d {
                    let product: f64 = (0..big_d).map(|k| p[i * big_d + k] * q[k * d + j]).sum();
                    assert_eq!(product, f64::from(i == j));
                }
            }
            let repeated = material(d, h, lambda, &[7; 32]);
            assert_eq!(p, repeated.0);
            assert_eq!(q, repeated.1);
        }
    }
}
