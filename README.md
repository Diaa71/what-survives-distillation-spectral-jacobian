# What Survives Distillation?

**Spectral Structure of Jacobian Eigenvectors in Consistency-Distilled Diffusion Models**

Manar Hamed, Diaa Azzam, Omid Reza Heidari

We investigate whether the geometry-adaptive spectral structure of diffusion model Jacobians survives consistency distillation. Analysing an EDM teacher and three consistency-distilled students sharing the same 295.9M-parameter UNet on ImageNet-64. 

## Models

| Model | Type | Steps | FID |
|-------|------|-------|-----|
| EDM Teacher | Multi-step diffusion (Karras et al., 2022) | 50 | 1.4 |
| CD-L2 | Consistency distillation, L2 loss | 1 | 4.7 |
| CD-LPIPS | Consistency distillation, LPIPS loss | 1 | 2.9 |
| CT | Consistency training from scratch | 1 | 11.1 |

All models share the same 295.9M-parameter class-conditional UNet architecture. Any spectral differences are process-dependent.

## Repository Structure

```
src/                          # Core library
  jacobian.py                 #   Halko randomised eigendecomposition
  bases.py                    #   Reference bases (DCT, edge Laplacian) and harmonicity
  basis_map.py                #   Teacher-student basis map M = V_S V_T^T
  eigenvalues.py              #   Effective rank and spectral analysis
  models/                     #   Model loading (EDM, consistency, Kadkhodaie)
  data.py                     #   ImageNet-64 data loading
  sampling.py                 #   EDM and consistency sampling

scripts/
  final_replication.py        #  30 images, all analyses


results/
  experiments/
    final_replication.json    #   All numerical results reported in the paper
  reproduction/
    phase_c_results.json      #   C^α PSNR scaling law validation
    phase_d_results.json      #   CelebA memorisation-to-generalisation validation
```

## Reproduction

### Pipeline Validation (Section 4)

- PSNR scaling law α/(α+1) across 5 regularity values (within 9% of theory)
- Jacobian eigenvector structure on C^α images (all 5 α values)


## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, scikit-image, matplotlib



## References

- Kadkhodaie, Z., Guth, F., Simoncelli, E. P., & Mallat, S. (2024). Generalization in diffusion models arises from geometry-adaptive harmonic representations. ICLR 2024.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. NeurIPS 2022.
- Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models. ICML 2023.
- Mallat, S., & Peyré, G. (2008). Orthogonal bandlet bases for geometric images approximation. CPAM 61(9).
