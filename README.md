# Lightweight Neural Basis Functions (NBF) — Minimal Reproduction

This repo implements a minimal reproduction of "Lightweight Neural Basis Functions for All-Frequency Shading" (SIGGRAPH Asia 2022). It trains a compact neural basis over the sphere to:
- Represent spherical signals (lighting, visibility-like, BRDF-like).
- Support fast Double/Triple Product Integrals (DPI/TPI) in coefficient space.

Key ideas:
- A small shared MLP outputs B basis channels per direction ω ∈ S².
- Any function f(ω) is projected to coefficients c via least squares on sample directions.
- Precompute G_ij = ∫ φ_i φ_j and T_ijk = ∫ φ_i φ_j φ_k to evaluate:
  - DPI: ∫ f g ≈ c_f^T G c_g
  - TPI: ∫ f g h ≈ Σ_{ijk} c_f[i] c_g[j] c_h[k] T_{ijk}
- Orthonormalization/whitening makes G ≈ I to stabilize and reduce runtime cost.

This is a simplified, research-oriented reproduction—not an engine integration. It focuses on correctness and clarity rather than peak performance.

## Environment

- Python 3.10+
- CUDA optional (recommended)
- PyTorch 2.1+

```
pip install -r requirements.txt
```

## Quick start

1) Train neural basis (B=32 by default):
```
python scripts/train_nbf.py --out_dir outputs/nbf_b32 --basis_dim 32 --epochs 10 --device cuda
```

2) Precompute integrals (G, T) with Monte Carlo on S²:
```
python scripts/precompute_integrals.py --ckpt outputs/nbf_b32/nbf.pt --out_dir outputs/nbf_b32 --num_samples 65536 --device cuda
```

3) Run demo: project random spherical signals, compare DPI/TPI via coeff-space vs MC ground-truth:
```
python scripts/demo_all_frequency_shading.py --ckpt outputs/nbf_b32/nbf.pt --precomp outputs/nbf_b32/precomp.npz --num_funcs 8 --num_samples_mc 131072 --device cuda
```

Artifacts:
- `outputs/nbf_b32/nbf.pt`: trained model
- `outputs/nbf_b32/whiten.npz`: whitening matrix A s.t. A G A^T ≈ I
- `outputs/nbf_b32/precomp.npz`: G, T in whitened basis

## What this reproduction includes

- A trainable neural basis `φ(ω) ∈ R^B` via small MLP with Fourier features.
- Synthetic data generators for spherical signals:
  - Spherical Gaussians (von Mises-Fisher-like)
  - Sharp visibility-like masks
  - BRDF-like lobes (Phong-like)
- Loss terms:
  - Reconstruction loss for random target functions
  - Optional DPI/TPI consistency loss
- Orthonormalization/whitening step post-training
- Monte Carlo precomputation for G, T
- Coefficient projection and fast DPI/TPI in coefficient space

## Notes and limitations

- T tensor scales as O(B^3). Keep B small (e.g., 16-48) unless you add tensor decompositions (CP/Tucker).
- Monte Carlo integration variance diminishes with larger sample counts; adjust `--num_samples` accordingly.
- This is a simplified reproduction for educational purposes. Production integration would:
  - Use richer datasets (HDR envmaps, real BRDFs, visibility from geometry)
  - Add better losses, curriculum, and numerical stabilizers
  - Use tensor decompositions and block-sparsity

## License

MIT
