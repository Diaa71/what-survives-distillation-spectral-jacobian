#!/usr/bin/env python3
"""
FINAL EXPERIMENT: Independent Replication + Basis Characterization
==================================================================

30 fresh images (indices 10-39) at σ=0.8  (primary)
10 of those at σ=1.5                       (σ-stability check)
Kadkhodaie C^α + CelebA models            (cross-architecture comparison)

Analyses:
  1. Cross-image eigenvector similarity
  2. Eigenvalue spectrum
  3. Teacher–student basis map + selective GAHB destruction
  4. DCT / Wavelet / PCA reference basis alignment
  5. Cross-architecture comparison (Kadkhodaie vs EDM)
  6. Eigenvector visualization grids
  7. Power spectrum of eigenvectors
  8. Rank-0 deep dive
  9. Geometry-adaptive reference bases (edge Laplacian + shearlets)
 10. σ=1.5 stability check


Outputs:
  results/experiments/final_replication.json
  results/experiments/figures/*.png
"""

import gc
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════
REPO = Path(__file__).resolve().parent.parent
if not (REPO / "src").is_dir():
    for cand in [Path("."), Path(".."),
                 Path("/home/diaa/IFT_project_what_survives_distillation")]:
        if (cand / "src").is_dir():
            REPO = cand.resolve()
            break
sys.path.insert(0, str(REPO))

import torch
from src.jacobian import halko_sym_eig
from src.bases import grayscale_collapse
from src.data import load_imagenet64_test
from src.models import load_edm_teacher, load_cd_student, make_denoise_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 30
P_OVER = 10
SEED_EIG = 42
TIKHONOV = 1e-6

# Primary analysis: 30 fresh images at σ=0.8
SIGMA_PRIMARY = 0.8
IMAGE_INDICES = list(range(10, 40))
N_IMAGES = len(IMAGE_INDICES)

# Secondary analysis: σ=1.5 on first 10 of those
SIGMA_SECONDARY = 1.5
SECONDARY_INDICES = IMAGE_INDICES[:10]
N_SECONDARY = len(SECONDARY_INDICES)

MODEL_NAMES = ["teacher", "cd_l2", "cd_lpips", "ct"]
STUDENT_NAMES = ["cd_l2", "cd_lpips", "ct"]
FIDS = {"teacher": 1.4, "cd_l2": 4.7, "cd_lpips": 2.9, "ct": 11.1}

H, W = 64, 64
d_full = 3 * H * W       # 12288
d_gs = H * W             # 4096
random_baseline = K / d_gs

CACHE_DIR = REPO / "results" / "experiments" / "cached_eigenvectors"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("FINAL REPLICATION: 30 Fresh Images + σ Stability + GAHB Test")
print("=" * 70)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Primary: {N_IMAGES} images at σ={SIGMA_PRIMARY}")
print(f"Secondary: {N_SECONDARY} images at σ={SIGMA_SECONDARY}")

test_imgs = load_imagenet64_test(
    REPO / "data" / "imagenet64" / "cached_data_imagenet64.pt"
)
print(f"Test set: {test_imgs.shape}")
assert test_imgs.shape[0] > max(IMAGE_INDICES)


# ═══════════════════════════════════════════════════════════════════
# HALKO EIGENDECOMPOSITION
# ═══════════════════════════════════════════════════════════════════
def run_halko_batch(sigma, image_indices, label=""):
    """Compute and cache Halko eigenvectors for all models on given images."""
    n = len(image_indices)
    total = len(MODEL_NAMES) * n
    done = 0
    t0 = time.time()

    eig_store = {}
    gs_store = {}

    for model_name in MODEL_NAMES:
        print(f"\n  === {model_name} ({label}) ===")
        model = load_edm_teacher(device=DEVICE) if model_name == "teacher" \
            else load_cd_student(model_name, device=DEVICE)
        denoise_fn = make_denoise_fn(model, sigma=sigma)

        eig_store[model_name] = {}
        gs_store[model_name] = {}

        for local_idx, global_idx in enumerate(image_indices):
            f_eigs = CACHE_DIR / f"{model_name}_img{global_idx}_sigma{sigma}_eigs.npy"
            f_evecs = CACHE_DIR / f"{model_name}_img{global_idx}_sigma{sigma}_evecs.npy"

            if f_eigs.exists() and f_evecs.exists():
                eigs = np.load(f_eigs)
                evecs = np.load(f_evecs)
                print(f"    img {global_idx}: CACHED λ₀={eigs[0]:.4f}")
            else:
                clean = test_imgs[global_idx:global_idx + 1].to(DEVICE)
                torch.manual_seed(SEED_EIG + global_idx)
                noisy = clean + sigma * torch.randn_like(clean)
                t1 = time.time()
                eigs, evecs = halko_sym_eig(
                    denoise_fn, noisy, K=K, p=P_OVER,
                    seed=SEED_EIG, device=DEVICE,
                    show_progress=False, tikhonov_eps_rel=TIKHONOV,
                )
                np.save(f_eigs, eigs)
                np.save(f_evecs, evecs)
                print(f"    img {global_idx}: λ₀={eigs[0]:.4f} ({time.time()-t1:.0f}s)")

            eig_store[model_name][local_idx] = {"eigs": eigs, "evecs": evecs}

            V_gs, _ = grayscale_collapse(evecs, C=3, H=H, W=W)
            for kk in range(K):
                nm = np.linalg.norm(V_gs[kk])
                if nm > 1e-12:
                    V_gs[kk] /= nm
            gs_store[model_name][local_idx] = V_gs

            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"      [{done}/{total}] ETA: {eta / 60:.0f} min")

        del model, denoise_fn
        free_gpu()

    print(f"\n  {label} done in {(time.time() - t0) / 60:.1f} min")
    return eig_store, gs_store


print("\n" + "=" * 70)
print("PHASE 1: Halko at σ=0.8 on 30 fresh images")
print("=" * 70)
t_start = time.time()
eig_data, gs_data = run_halko_batch(SIGMA_PRIMARY, IMAGE_INDICES, label="σ=0.8")


print("\n" + "=" * 70)
print("PHASE 2: Halko at σ=1.5 on 10 images (stability check)")
print("=" * 70)
eig_data_s2, gs_data_s2 = run_halko_batch(SIGMA_SECONDARY, SECONDARY_INDICES, label="σ=1.5")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Cross-Image Eigenvector Similarity
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 1: Cross-Image Eigenvector Similarity (σ=0.8, 30 images)")
print("=" * 70)

cos_random = float(np.sqrt(2 / (np.pi * d_full)))
pairs = list(combinations(range(N_IMAGES), 2))

def compute_cross_sim(eig_store, n_imgs, pair_list):
    results = {}
    for m in MODEL_NAMES:
        per_rank = []
        for k in range(K):
            sims = []
            for i, j in pair_list:
                vi = eig_store[m][i]["evecs"][k]
                vj = eig_store[m][j]["evecs"][k]
                cos = abs(float(np.dot(vi / np.linalg.norm(vi),
                                        vj / np.linalg.norm(vj))))
                sims.append(cos)
            per_rank.append(float(np.mean(sims)))
        top8 = float(np.mean(per_rank[:8]))
        results[m] = {
            "top8_cos": top8,
            "over_random": top8 / cos_random,
            "per_rank_cos": per_rank,
        }
    return results

cross_sim = compute_cross_sim(eig_data, N_IMAGES, pairs)
teacher_adaptivity = np.array(cross_sim["teacher"]["per_rank_cos"])

for m in MODEL_NAMES:
    print(f"  {m}: {cross_sim[m]['over_random']:.1f}× random")

print(f"\n  Teacher / Student gap: {cross_sim['teacher']['over_random']:.1f}× vs "
      f"{np.mean([cross_sim[s]['over_random'] for s in STUDENT_NAMES]):.1f}×")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Eigenvalue Spectrum
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 2: Eigenvalue Spectrum")
print("=" * 70)

eigenvalue_results = {}
for m in MODEL_NAMES:
    l0s, reffs = [], []
    for i in range(N_IMAGES):
        eigs = eig_data[m][i]["eigs"]
        l0s.append(float(eigs[0]))
        ep = np.maximum(eigs, 0)
        s, s2 = np.sum(ep), np.sum(ep ** 2)
        reffs.append(float(s ** 2 / s2) if s2 > 0 else 0)
    eigenvalue_results[m] = {"mean_lambda0": float(np.mean(l0s)),
                              "mean_reff": float(np.mean(reffs))}
    print(f"  {m}: λ₀ = {np.mean(l0s):.3f}, r_eff = {np.mean(reffs):.1f}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Basis Map + Selective GAHB Destruction
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 3: Basis Map + Selective GAHB Destruction")
print("=" * 70)

from scipy.stats import spearmanr

basis_map_results = {}
rank0_survivals = {s: [] for s in STUDENT_NAMES}

for sn in STUDENT_NAMES:
    overlaps, diags = [], []
    survival = np.zeros(K)
    for i in range(N_IMAGES):
        M = gs_data[sn][i] @ gs_data["teacher"][i].T
        overlaps.append(float(np.sum(M ** 2) / K))
        de = float(np.sum(np.diag(M) ** 2))
        te = float(np.sum(M ** 2))
        diags.append(de / te if te > 0 else 0)
        s_k = np.sum(M ** 2, axis=0)
        survival += s_k
        rank0_survivals[sn].append(float(s_k[0]))
    survival /= N_IMAGES

    rho, p = spearmanr(survival, teacher_adaptivity)
    si = np.argsort(teacher_adaptivity)
    P_adapt = float(np.mean(survival[si[:K // 2]]))
    P_generic = float(np.mean(survival[si[K // 2:]]))

    basis_map_results[sn] = {
        "mean_overlap": float(np.mean(overlaps)),
        "mean_diag": float(np.mean(diags)),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "P_adaptive": P_adapt, "P_generic": P_generic,
        "ratio": P_generic / P_adapt if P_adapt > 0 else 0,
        "survival_per_rank": survival.tolist(),
    }
    print(f"  {sn}: O={np.mean(overlaps):.3f}, ρ={rho:.3f} (p={p:.4f}), "
          f"generic/adaptive={P_generic / P_adapt:.2f}×")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 4: Reference Basis Alignment (DCT, Wavelet, PCA)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 4: Reference Basis Alignment")
print("=" * 70)

def build_dct_basis(Hb, Wb, n):
    freqs = sorted([(u**2 + v**2, u, v) for u in range(Hb) for v in range(Wb)
                    if not (u == 0 and v == 0)])
    basis = []
    x, y = np.arange(Hb), np.arange(Wb)
    for _, u, v in freqs[:n]:
        vec = np.outer(np.cos(np.pi * (2*x+1) * u / (2*Hb)),
                       np.cos(np.pi * (2*y+1) * v / (2*Wb))).flatten()
        vec /= np.linalg.norm(vec)
        basis.append(vec)
    return np.array(basis)

DCT_BASIS = build_dct_basis(H, W, K)

WAV_AVAILABLE = False
try:
    import pywt
    def build_wavelet_basis(Hb, Wb, n, wavelet="db4"):
        dummy = np.zeros((Hb, Wb))
        max_lev = pywt.dwt_max_level(min(Hb, Wb), pywt.Wavelet(wavelet).dec_len)
        coeffs = pywt.wavedec2(dummy, wavelet, level=max_lev)
        basis = []
        for li in range(len(coeffs)):
            if li == 0:
                arr = coeffs[0]
                for ii in range(arr.shape[0]):
                    for jj in range(arr.shape[1]):
                        zc = [np.zeros_like(coeffs[0])]
                        for lv in range(1, len(coeffs)):
                            zc.append(tuple(np.zeros_like(d) for d in coeffs[lv]))
                        zc[0][ii, jj] = 1.0
                        vec = pywt.waverec2(zc, wavelet)[:Hb, :Wb].flatten()
                        nm = np.linalg.norm(vec)
                        if nm > 1e-12: vec /= nm
                        basis.append(vec)
            else:
                for di in range(3):
                    arr = coeffs[li][di]
                    for ii in range(arr.shape[0]):
                        for jj in range(arr.shape[1]):
                            zc = [np.zeros_like(coeffs[0])]
                            for lv in range(1, len(coeffs)):
                                zc.append(tuple(np.zeros_like(d) for d in coeffs[lv]))
                            dl = list(zc[li]); dl[di][ii, jj] = 1.0; zc[li] = tuple(dl)
                            vec = pywt.waverec2(zc, wavelet)[:Hb, :Wb].flatten()
                            nm = np.linalg.norm(vec)
                            if nm > 1e-12: vec /= nm
                            basis.append(vec)
            if len(basis) >= n: break
        return np.array(basis[:n])
    WAV_BASIS = build_wavelet_basis(H, W, K)
    WAV_AVAILABLE = True
    print(f"  Wavelet basis: {WAV_BASIS.shape}")
except ImportError:
    print("  PyWavelets not available — skipping wavelet")

imgs_gs_arr = np.array([test_imgs[IMAGE_INDICES[i]].numpy().mean(axis=0).flatten()
                         for i in range(N_IMAGES)])
U_pca, S_pca, Vt_pca = np.linalg.svd(imgs_gs_arr - imgs_gs_arr.mean(axis=0), full_matrices=False)
PCA_BASIS = Vt_pca[:K]
print(f"  PCA basis: {min(N_IMAGES, K)} meaningful components")

ref_bases = {"dct": DCT_BASIS}
if WAV_AVAILABLE: ref_bases["wavelet"] = WAV_BASIS
ref_bases["pca"] = PCA_BASIS

alignment_results = {}
dct_per_image = {m: [] for m in MODEL_NAMES}

for ref_name, ref_b in ref_bases.items():
    alignment_results[ref_name] = {}
    for m in MODEL_NAMES:
        vals = []
        for i in range(N_IMAGES):
            Mr = gs_data[m][i] @ ref_b.T
            ov = float(np.sum(Mr ** 2) / K)
            vals.append(ov)
            if ref_name == "dct":
                dct_per_image[m].append(ov)
        alignment_results[ref_name][m] = {"mean": float(np.mean(vals)),
                                           "std": float(np.std(vals))}

print(f"\n  {'Model':<12} {'FID':<6}", end="")
for r in ref_bases: print(f" {r:<12}", end="")
print(f" {'cross-sim':<12}")
print("  " + "-" * (18 + 12 * (len(ref_bases) + 1)))
for m in MODEL_NAMES:
    print(f"  {m:<12} {FIDS[m]:<6.1f}", end="")
    for r in ref_bases: print(f" {alignment_results[r][m]['mean']:<12.4f}", end="")
    print(f" {cross_sim[m]['over_random']:<12.1f}×")

teacher_per_rank_dct = np.zeros(K)
for i in range(N_IMAGES):
    Md = gs_data["teacher"][i] @ DCT_BASIS.T
    teacher_per_rank_dct += np.sum(Md ** 2, axis=1)
teacher_per_rank_dct /= N_IMAGES

rho_dct_adapt, p_dct_adapt = spearmanr(teacher_per_rank_dct, teacher_adaptivity)
print(f"\n  ρ(teacher DCT, adaptivity) = {rho_dct_adapt:.3f} (p={p_dct_adapt:.4f})")

teacher_wins = sum(1 for i in range(N_IMAGES)
                   if dct_per_image["teacher"][i] > max(dct_per_image[s][i] for s in STUDENT_NAMES))
print(f"  Teacher > best student on {teacher_wins}/{N_IMAGES} images (DCT)")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 5: Kadkhodaie Cross-Architecture Comparison
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 5: Kadkhodaie Models vs EDM")
print("=" * 70)

kadkhodaie_results = {}
N_KAD = 5
CA_OK = False
ca_evecs_gs = {}

try:
    from src.models import load_kadkhodaie_unet, make_denoiser, get_c_alpha_ckpt
    from src.data import generate_c_alpha_batch

    SIGMA_CA = 0.25
    ckpt = get_c_alpha_ckpt(3, root=str(REPO / "checkpoints" / "kadkhodaie"))
    model_ca = load_kadkhodaie_unet(ckpt, device=DEVICE)
    dn_ca = make_denoiser(model_ca, DEVICE)

    ca_eigs = {}
    H_CA, W_CA = 80, 80; d_ca = H_CA * W_CA

    for idx in range(N_KAD):
        batch = generate_c_alpha_batch(alpha=3, n=200, seed=idx * 100)
        clean = batch[9:10].to(DEVICE)
        torch.manual_seed(SEED_EIG + idx)
        noisy = clean + SIGMA_CA * torch.randn_like(clean)
        eigs, evecs = halko_sym_eig(dn_ca, noisy, K=K, p=P_OVER,
                                     seed=SEED_EIG, device=DEVICE,
                                     show_progress=False, tikhonov_eps_rel=0.0)
        vgs = evecs.copy() if evecs.shape[1] == d_ca else grayscale_collapse(evecs, C=1, H=H_CA, W=W_CA)[0]
        for kk in range(K):
            nm = np.linalg.norm(vgs[kk])
            if nm > 1e-12: vgs[kk] /= nm
        ca_evecs_gs[idx] = vgs; ca_eigs[idx] = eigs
        print(f"  C^α img {idx}: λ₀={eigs[0]:.3f}")

    cr = float(np.sqrt(2 / (np.pi * d_ca)))
    ca_top8 = float(np.mean([
        np.mean([abs(float(np.dot(ca_evecs_gs[i][k], ca_evecs_gs[j][k])))
                 for i, j in combinations(range(N_KAD), 2)])
        for k in range(8)]))

    DCT_80 = build_dct_basis(H_CA, W_CA, K)
    ca_dct = float(np.mean([np.sum((ca_evecs_gs[i] @ DCT_80.T) ** 2) / K for i in range(N_KAD)]))

    kadkhodaie_results["c_alpha_3"] = {
        "cross_sim": ca_top8 / cr, "dct_alignment": ca_dct,
        "dct_times_random": ca_dct / (K / d_ca),
        "mean_lambda0": float(np.mean([ca_eigs[i][0] for i in range(N_KAD)])),
    }
    print(f"  C^α: cross-sim={ca_top8/cr:.1f}×, DCT={ca_dct/(K/d_ca):.1f}×")
    del model_ca, dn_ca; free_gpu()
    CA_OK = True
except Exception as e:
    print(f"  C^α failed: {e}")

# CelebA
try:
    from src.models import get_celeba_ckpt
    for N_cel in [1, 1000]:
        try:
            ckpt = get_celeba_ckpt(N_cel, root=str(REPO / "checkpoints" / "kadkhodaie"))
            model_cel = load_kadkhodaie_unet(ckpt, device=DEVICE)
            dn_cel = make_denoiser(model_cel, DEVICE)
            import glob as gl
            pt_files = gl.glob(str(REPO / "data" / "celeba" / "*.pt"))
            if not pt_files:
                print(f"  CelebA N={N_cel}: no data"); del model_cel, dn_cel; free_gpu(); continue
            cel_imgs = torch.load(pt_files[0])[:N_KAD]
            H_CEL, W_CEL = 80, 80; d_cel = H_CEL * W_CEL
            cel_gs = {}; cel_eigs_s = {}
            for idx in range(min(N_KAD, cel_imgs.shape[0])):
                c = cel_imgs[idx:idx+1].to(DEVICE)
                torch.manual_seed(SEED_EIG + idx)
                ny = c + 0.15 * torch.randn_like(c)
                eigs, evecs = halko_sym_eig(dn_cel, ny, K=K, p=P_OVER, seed=SEED_EIG,
                                             device=DEVICE, show_progress=False, tikhonov_eps_rel=0.0)
                vgs = evecs.copy() if evecs.shape[1] == d_cel else grayscale_collapse(evecs, C=1, H=H_CEL, W=W_CEL)[0]
                for kk in range(K):
                    nm = np.linalg.norm(vgs[kk])
                    if nm > 1e-12: vgs[kk] /= nm
                cel_gs[idx] = vgs; cel_eigs_s[idx] = eigs
            cr_cel = float(np.sqrt(2 / (np.pi * d_cel)))
            cel_top8 = float(np.mean([np.mean([abs(float(np.dot(cel_gs[i][k], cel_gs[j][k])))
                     for i, j in combinations(range(len(cel_gs)), 2)]) for k in range(8)]))
            DCT_CEL = build_dct_basis(H_CEL, W_CEL, K)
            cel_dct = float(np.mean([np.sum((cel_gs[i] @ DCT_CEL.T) ** 2) / K for i in range(len(cel_gs))]))
            kadkhodaie_results[f"celeba_N{N_cel}"] = {
                "cross_sim": cel_top8 / cr_cel, "dct_times_random": cel_dct / (K / d_cel),
                "mean_lambda0": float(np.mean([cel_eigs_s[i][0] for i in range(len(cel_gs))])),
            }
            print(f"  CelebA N={N_cel}: cross-sim={cel_top8/cr_cel:.1f}×, DCT={cel_dct/(K/d_cel):.1f}×")
            del model_cel, dn_cel; free_gpu()
        except Exception as e:
            print(f"  CelebA N={N_cel}: {e}")
except ImportError:
    pass

# Unified table
print(f"\n  {'Model':<22} {'Arch':<12} {'DCT (×r)':<12} {'Cross-sim':<12}")
print("  " + "-" * 58)
for key, label in [("c_alpha_3", "C^α (α=3)"), ("celeba_N1", "CelebA N=1"), ("celeba_N1000", "CelebA N=1000")]:
    if key in kadkhodaie_results:
        r = kadkhodaie_results[key]
        print(f"  {label:<22} {'Bias-free':<12} {r['dct_times_random']:<12.1f} {r.get('cross_sim',0):<12.1f}")
print("  " + "- " * 29)
for m in MODEL_NAMES:
    print(f"  {m:<22} {'EDM UNet':<12} {alignment_results['dct'][m]['mean']/random_baseline:<12.1f} "
          f"{cross_sim[m]['over_random']:<12.1f}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 6-8: Visualizations
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSES 6-8: Visualizations")
print("=" * 70)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Eigenvector grids (3 images)
for sl in [0, 10, 20]:
    if sl >= N_IMAGES: continue
    gi = IMAGE_INDICES[sl]
    fig, axes = plt.subplots(len(MODEL_NAMES)+1, 9, figsize=(27, 3*(len(MODEL_NAMES)+1)))
    cl = test_imgs[gi].numpy().transpose(1,2,0)
    cl = np.clip(cl, 0, 1) if cl.max() <= 1 else np.clip(cl/255, 0, 1)
    axes[0,0].imshow(cl); axes[0,0].set_title(f"Image {gi}"); axes[0,0].axis("off")
    for c in range(1,9): axes[0,c].axis("off")
    for row, m in enumerate(MODEL_NAMES):
        eigs = eig_data[m][sl]["eigs"]; evecs = eig_data[m][sl]["evecs"]
        axes[row+1,0].imshow(cl); axes[row+1,0].set_title(m, fontweight="bold"); axes[row+1,0].axis("off")
        for k in range(8):
            v = evecs[k].reshape(3,H,W).mean(axis=0); vm = max(abs(v.min()),abs(v.max()),1e-8)
            axes[row+1,k+1].imshow(v, cmap="RdBu", vmin=-vm, vmax=vm)
            axes[row+1,k+1].set_title(f"e{k} λ={eigs[k]:.3f}", fontsize=7); axes[row+1,k+1].axis("off")
    plt.suptitle(f"Eigenvectors — image {gi} (σ={SIGMA_PRIMARY})", fontsize=13)
    plt.tight_layout(); plt.savefig(FIG_DIR/f"eigvec_grid_img{gi}.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved eigvec_grid_img{gi}.png")

# Power spectrum
n_bins = 16
def radial_power(v2d, nb=16):
    fft = np.fft.fft2(v2d); pw = np.abs(fft)**2
    h,w = v2d.shape; cy,cx = h//2, w//2
    r = np.fft.fftshift(np.sqrt((np.arange(w)[None,:]-cx)**2+(np.arange(h)[:,None]-cy)**2))
    bins = np.linspace(0, min(cy,cx), nb+1)
    return np.array([pw[(r>=bins[b])&(r<bins[b+1])].mean() if ((r>=bins[b])&(r<bins[b+1])).any() else 0 for b in range(nb)])

power_spectra = {}
for m in MODEL_NAMES:
    power_spectra[m] = np.mean([radial_power(gs_data[m][i][k].reshape(H,W)) for i in range(N_IMAGES) for k in range(8)], axis=0)
if CA_OK:
    power_spectra["c_alpha"] = np.mean([radial_power(ca_evecs_gs[i][k].reshape(H_CA,W_CA)) for i in range(N_KAD) for k in range(8)], axis=0)

fig, ax = plt.subplots(figsize=(10,6))
for m in MODEL_NAMES: ax.plot(power_spectra[m]/power_spectra[m].max(), label=m, lw=2)
if "c_alpha" in power_spectra: ax.plot(power_spectra["c_alpha"]/power_spectra["c_alpha"].max(), label="C^α", lw=2, ls="--")
ax.set_xlabel("Radial frequency"); ax.set_ylabel("Normalized power"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR/"power_spectrum.png", dpi=150); plt.close()
print("  Saved power_spectrum.png")

# Rank-0 deep dive
n_show = min(10, N_IMAGES)
fig, axes = plt.subplots(n_show, 5, figsize=(15, 3*n_show))
for i in range(n_show):
    gi = IMAGE_INDICES[i]; cl = test_imgs[gi].numpy().transpose(1,2,0)
    cl = np.clip(cl, 0, 1) if cl.max() <= 1 else np.clip(cl/255, 0, 1)
    axes[i,0].imshow(cl); axes[i,0].set_title(f"img {gi}", fontsize=8); axes[i,0].axis("off")
    tv = gs_data["teacher"][i][0].reshape(H,W); vm = max(abs(tv.min()),abs(tv.max()),1e-8)
    axes[i,1].imshow(tv, cmap="RdBu", vmin=-vm, vmax=vm)
    axes[i,1].set_title(f"Teacher e₀\nλ={eig_data['teacher'][i]['eigs'][0]:.3f}", fontsize=8); axes[i,1].axis("off")
    for si, sn in enumerate(STUDENT_NAMES):
        sv = gs_data[sn][i][0].reshape(H,W); vm2 = max(abs(sv.min()),abs(sv.max()),1e-8)
        Mb = gs_data[sn][i] @ gs_data["teacher"][i].T; s0 = float(np.sum(Mb[:,0]**2))
        axes[i,2+si].imshow(sv, cmap="RdBu", vmin=-vm2, vmax=vm2)
        axes[i,2+si].set_title(f"{sn} e₀\nsurv={s0:.2f}", fontsize=7); axes[i,2+si].axis("off")
plt.suptitle("Rank-0: teacher's most adaptive direction vs students", fontsize=12)
plt.tight_layout(); plt.savefig(FIG_DIR/"rank0_deep_dive.png", dpi=150, bbox_inches="tight"); plt.close()
print("  Saved rank0_deep_dive.png")
for sn in STUDENT_NAMES:
    print(f"  {sn}: rank-0 survival = {np.mean(rank0_survivals[sn]):.3f}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 9: GAHB Reference Bases (Edge Laplacian + Shearlets)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 9: Geometry-Adaptive Reference Bases")
print("=" * 70)

gahb_test_results = {}

# Part A: bilateral edge-adaptive Laplacian
print("\n  Part A: Edge-adaptive Laplacian")
LAP_OK = False; lap_results = {}
try:
    from scipy.sparse import lil_matrix, diags as sp_diags
    from scipy.sparse.linalg import eigsh

    def build_bilateral_lap(img_gs, K_b, h_int):
        Hb, Wb = img_gs.shape; N = Hb*Wb; flat = img_gs.flatten()
        W_sp = lil_matrix((N,N), dtype=np.float64)
        for i in range(Hb):
            for j in range(Wb):
                idx = i*Wb+j
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ni,nj = i+di, j+dj
                    if 0<=ni<Hb and 0<=nj<Wb:
                        nidx = ni*Wb+nj
                        W_sp[idx,nidx] = np.exp(-(flat[idx]-flat[nidx])**2/(2*h_int**2))
        W_sp = W_sp.tocsr()
        D = np.maximum(np.array(W_sp.sum(axis=1)).flatten(), 1e-12)
        Disq = sp_diags(1.0/np.sqrt(D))
        L = sp_diags(np.ones(N)) - Disq @ W_sp @ Disq
        vals, vecs = eigsh(L, k=K_b+1, which='SM', maxiter=5000, tol=1e-6)
        order = np.argsort(vals); basis = vecs[:,order[1:K_b+1]].T
        for kk in range(basis.shape[0]):
            nm = np.linalg.norm(basis[kk])
            if nm > 1e-12: basis[kk] /= nm
        return basis

    # h from median neighbor diff
    sd = []
    for i in range(min(5,N_IMAGES)):
        ig = test_imgs[IMAGE_INDICES[i]].numpy().mean(axis=0).flatten()
        for r in range(H):
            for c in range(W):
                if c+1<W: sd.append(abs(ig[r*W+c]-ig[r*W+c+1]))
                if r+1<H: sd.append(abs(ig[r*W+c]-ig[(r+1)*W+c]))
    h_med = float(np.median(sd))
    H_VALS = [h_med*0.5, h_med, h_med*2.0]
    print(f"  h values: {[f'{h:.4f}' for h in H_VALS]}")

    lap_align = {hi: {} for hi in range(3)}
    for hi, hv in enumerate(H_VALS):
        for i in range(N_IMAGES):
            ig = test_imgs[IMAGE_INDICES[i]].numpy().mean(axis=0)
            try:
                lb = build_bilateral_lap(ig, K, hv)
                for m in MODEL_NAMES:
                    Mg = gs_data[m][i] @ lb.T
                    lap_align[hi].setdefault(m,[]).append(float(np.sum(Mg**2)/K))
            except: pass

    print(f"\n  {'Model':<12}", end="")
    for hv in H_VALS: print(f" h={hv:.3f}{'':>5}", end="")
    print()
    for m in MODEL_NAMES:
        print(f"  {m:<12}", end="")
        for hi in range(3):
            if m in lap_align[hi]:
                print(f" {np.mean(lap_align[hi][m])/random_baseline:<14.1f}×", end="")
            else: print(f" {'N/A':<14}", end="")
        print()
        if m in lap_align[1]: lap_results[m] = float(np.mean(lap_align[1][m]))

    gahb_test_results["edge_laplacian"] = {m: {"mean": lap_results.get(m,0),
        "times_random": lap_results.get(m,0)/random_baseline} for m in MODEL_NAMES}
    LAP_OK = True
except Exception as e:
    print(f"  Failed: {e}")

# Part B: Shearlet sparsity
print("\n  Part B: Shearlet sparsity")
SHEARLET_OK = False; shearlet_results = {}
try:
    from pyshearlab import SLgetShearletSystem2D, SLsheardec2D
    shsys = SLgetShearletSystem2D(0, H, W, 4)
    M_top = shsys['nShearlets'] // 4
    print(f"  {shsys['nShearlets']} shearlets, top-25% energy")

    for m in MODEL_NAMES:
        sp = []; 
        for i in range(N_IMAGES):
            for k in range(8):
                v = gs_data[m][i][k].reshape(H,W)
                coeffs = SLsheardec2D(v.astype(np.float64), shsys)
                ac = np.abs(coeffs.flatten()); acs = np.sort(ac)[::-1]
                te = np.sum(ac**2)
                sp.append(float(np.sum(acs[:M_top]**2)/te) if te > 0 else 0)
        shearlet_results[m] = float(np.mean(sp))
        print(f"  {m}: {np.mean(sp):.4f}")

    if CA_OK:
        shsys80 = SLgetShearletSystem2D(0, H_CA, W_CA, 4); Mt80 = shsys80['nShearlets']//4
        ca_sp = []
        for i in range(N_KAD):
            for k in range(8):
                v = ca_evecs_gs[i][k].reshape(H_CA,W_CA)
                coeffs = SLsheardec2D(v.astype(np.float64), shsys80)
                ac = np.abs(coeffs.flatten()); acs = np.sort(ac)[::-1]; te = np.sum(ac**2)
                ca_sp.append(float(np.sum(acs[:Mt80]**2)/te) if te > 0 else 0)
        shearlet_results["c_alpha"] = float(np.mean(ca_sp))
        print(f"  C^α: {np.mean(ca_sp):.4f}")
    gahb_test_results["shearlet"] = shearlet_results; SHEARLET_OK = True
except ImportError:
    print("  pyshearlab not installed")
except Exception as e:
    print(f"  Failed: {e}")

# Visualization
if LAP_OK:
    try:
        ig = test_imgs[IMAGE_INDICES[0]].numpy().mean(axis=0)
        flat = ig.flatten()
        N_pix = H * W

        import scipy.sparse as sp_mod
        W_dense = np.zeros((N_pix, N_pix))
        for i_px in range(H):
            for j_px in range(W):
                idx = i_px * W + j_px
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ni, nj = i_px+di, j_px+dj
                    if 0 <= ni < H and 0 <= nj < W:
                        nidx = ni * W + nj
                        W_dense[idx, nidx] = np.exp(-(flat[idx]-flat[nidx])**2/(2*h_med**2))

        D_vec = W_dense.sum(axis=1)
        D_isq = np.diag(1.0 / np.sqrt(np.maximum(D_vec, 1e-12)))
        L_dense = np.eye(N_pix) - D_isq @ W_dense @ D_isq

        eigenvalues_lap, eigenvectors_lap = np.linalg.eigh(L_dense)
        lap_show = eigenvectors_lap[:, 1:9].T
        for kk in range(8):
            nm = np.linalg.norm(lap_show[kk])
            if nm > 1e-12: lap_show[kk] /= nm

        fig, axes = plt.subplots(3, 9, figsize=(27,9))
        cl = test_imgs[IMAGE_INDICES[0]].numpy().transpose(1,2,0)
        cl = np.clip(cl, 0, 1) if cl.max() <= 1 else np.clip(cl/255, 0, 1)
        for row, (lbl, vcs) in enumerate([
            ("Edge-Laplacian\n(GAHB ref)", lap_show),
            ("Teacher", gs_data["teacher"][0][:8]),
            ("CD-LPIPS", gs_data["cd_lpips"][0][:8])]):
            axes[row,0].imshow(cl)
            axes[row,0].set_title(lbl, fontsize=9, fontweight="bold")
            axes[row,0].axis("off")
            for k in range(8):
                v = vcs[k].reshape(H,W)
                vm = max(abs(v.min()),abs(v.max()),1e-8)
                axes[row,k+1].imshow(v, cmap="RdBu", vmin=-vm, vmax=vm)
                axes[row,k+1].set_title(f"e{k}", fontsize=8)
                axes[row,k+1].axis("off")
        plt.suptitle("GAHB reference vs Teacher vs Student", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG_DIR/"gahb_vs_teacher_vs_student.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved gahb_vs_teacher_vs_student.png")
    except Exception as e:
        print(f"  GAHB visualization failed again: {e}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 10: σ=1.5 Stability Check
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 10: σ=1.5 Stability Check")
print("=" * 70)

pairs_s2 = list(combinations(range(N_SECONDARY), 2))
cross_sim_s2 = compute_cross_sim(eig_data_s2, N_SECONDARY, pairs_s2)

for m in MODEL_NAMES:
    v08 = cross_sim[m]["over_random"]; v15 = cross_sim_s2[m]["over_random"]
    ok = "YES" if (v08 < 15) == (v15 < 15) else "NO"
    print(f"  {m}: σ=0.8→{v08:.1f}×, σ=1.5→{v15:.1f}×  {ok}")


# ═══════════════════════════════════════════════════════════════════
# COMPARISON + VERDICT + SAVE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARISON WITH ORIGINALS")
print("=" * 70)
for label, path in [("exp9", REPO/"results"/"experiments"/"exp9_final_comparison.json"),
                     ("exp13", REPO/"results"/"experiments"/"exp13_basis_analysis.json")]:
    if path.exists():
        orig = json.load(open(path))
        if label == "exp9":
            print(f"  Cross-sim: ", end="")
            for m in MODEL_NAMES:
                ov = orig["models"][m]["0.8"]["cos_over_random"]; fv = cross_sim[m]["over_random"]
                print(f"{m}={ov:.0f}→{fv:.0f}  ", end="")
            print()
        elif label == "exp13":
            print(f"  DCT: ", end="")
            for m in MODEL_NAMES:
                ov = orig["alignment"]["dct"][m]["mean"]; fv = alignment_results["dct"][m]["mean"]
                print(f"{m}={ov:.4f}→{fv:.4f}  ", end="")
            print()

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

t_r = cross_sim["teacher"]["over_random"]
s_rs = [cross_sim[s]["over_random"] for s in STUDENT_NAMES]
test1 = t_r < min(s_rs)
print(f"\n  1. Adaptivity gap: teacher={t_r:.1f}× vs students={min(s_rs):.1f}–{max(s_rs):.1f}×  →  {'REPLICATES' if test1 else 'FAILS'}")

rhos = [basis_map_results[s]["spearman_rho"] for s in STUDENT_NAMES]
ps_bm = [basis_map_results[s]["spearman_p"] for s in STUDENT_NAMES]
test2 = all(r > 0.3 and p < 0.05 for r, p in zip(rhos, ps_bm))
rats = [basis_map_results[s]["ratio"] for s in STUDENT_NAMES]
print(f"  2. Selective destruction: ρ={np.mean(rhos):.3f}, ratio={np.mean(rats):.2f}×  →  {'REPLICATES' if test2 else 'FAILS'}")

t_dx = alignment_results["dct"]["teacher"]["mean"]/random_baseline
s_dx = max(alignment_results["dct"][s]["mean"]/random_baseline for s in STUDENT_NAMES)
test3 = t_dx > s_dx
print(f"  3. DCT alignment: teacher={t_dx:.1f}× vs student={s_dx:.1f}×, wins {teacher_wins}/{N_IMAGES}  →  {'REPLICATES' if test3 else 'FAILS'}")

all_pass = test1 and test2 and test3
print(f"\n  {'='*50}")
print(f"  ALL THREE CORE FINDINGS REPLICATE: {'YES' if all_pass else 'NO'}")
print(f"  {'='*50}")

results = {
    "experiment": "final_replication", "image_indices": IMAGE_INDICES,
    "sigma_primary": SIGMA_PRIMARY, "sigma_secondary": SIGMA_SECONDARY,
    "n_primary": N_IMAGES, "n_secondary": N_SECONDARY,
    "cross_sim_08": cross_sim, "cross_sim_15": cross_sim_s2,
    "eigenvalues": eigenvalue_results, "basis_map": basis_map_results,
    "ref_alignment": {r: {m: alignment_results[r][m] for m in MODEL_NAMES} for r in ref_bases},
    "kadkhodaie": kadkhodaie_results, "gahb_test": gahb_test_results,
    "power_low_freq": {m: float(power_spectra[m][:n_bins//2].sum()/power_spectra[m].sum()) for m in MODEL_NAMES},
    "rank0_survival": {s: float(np.mean(rank0_survivals[s])) for s in STUDENT_NAMES},
    "teacher_adaptivity": teacher_adaptivity.tolist(),
    "teacher_per_rank_dct": teacher_per_rank_dct.tolist(),
    "rho_dct_adapt": float(rho_dct_adapt),
    "verdict": {"test1": test1, "test2": test2, "test3": test3, "all_pass": all_pass},
}
json.dump(results, open(OUT_DIR/"final_replication.json", "w"), indent=2)
print(f"\nSaved final_replication.json")
print(f"\nFigures:"); import glob
for f in sorted(glob.glob(str(FIG_DIR/"*.png"))): print(f"  {f}")
print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
print("\n*** DONE ***")
