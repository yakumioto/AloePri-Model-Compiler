use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

fn rng(seed: &[u8; 32], domain: &str) -> ChaCha20Rng {
    let mut hash = blake3::Hasher::new();
    hash.update(b"aloepri-algorithm1-signed-null-v3-substream");
    hash.update(seed);
    hash.update(domain.as_bytes());
    ChaCha20Rng::from_seed(*hash.finalize().as_bytes())
}

fn gaussian(rows: usize, cols: usize, seed: &[u8; 32], domain: &str, scale: f64) -> Vec<f64> {
    let mut stream = rng(seed, domain);
    (0..rows * cols)
        .map(|_| {
            let value: f64 = StandardNormal.sample(&mut stream);
            value * scale
        })
        .collect()
}

pub(crate) fn material(d: usize, h: usize, lambda: f64, seed: &[u8; 32]) -> (Vec<f64>, Vec<f64>) {
    let rank = h / 2;
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
    let c = gaussian(d, rank, seed, "Cg", scale);
    let e = gaussian(d, rank, seed, "Eg", scale);
    let f = gaussian(rank, d, seed, "Fg", scale);
    let n = gaussian(rank, d, seed, "Ng", scale);
    let mut p = vec![0.0; d * big_d];
    let mut q = vec![0.0; big_d * d];
    for i in 0..d {
        p[i * big_d + i] = core[i];
        q[i * d + i] = core[i];
        for j in 0..rank {
            p[i * big_d + d + permutation[rank + j]] = c[i * rank + j] * aux_signs[rank + j];
            p[i * big_d + d + permutation[h + j]] = e[i * rank + j] * aux_signs[h + j];
            q[(d + permutation[j]) * d + i] = f[j * d + i] * aux_signs[j];
            q[(d + permutation[h + rank + j]) * d + i] = n[j * d + i] * aux_signs[h + rank + j];
        }
    }
    (p, q)
}
