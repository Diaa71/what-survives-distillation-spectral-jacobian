"""Loaders for the EDM teacher and CD/CT student family on ImageNet-64.

EDM preconditioning (Karras et al. 2022, eq. 7):
    c_skip = sigma_data^2 / ((sigma - sigma_min)^2 + sigma_data^2)
    c_out  = (sigma - sigma_min) * sigma_data / sqrt(sigma^2 + sigma_data^2)
    c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
    denoised = c_skip * y + c_out * model(c_in * y, t(sigma))

No .clamp(-1, 1) on output -- clamping zeros out Jacobian gradients.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
from diffusers import UNet2DModel


MODEL_IDS = {
    "teacher":  "dg845/diffusers-cm_edm_imagenet64_ema",
    "cd_l2":    "openai/diffusers-cd_imagenet64_l2",
    "cd_lpips": "openai/diffusers-cd_imagenet64_lpips",
    "ct":       "openai/diffusers-ct_imagenet64",
}

MODEL_COLORS = {
    "teacher": "#1f77b4", "cd_l2": "#d62728",
    "cd_lpips": "#ff7f0e", "ct": "#2ca02c",
}

MODEL_LABELS = {
    "teacher": "Teacher (EDM)", "cd_l2": "CD-L2",
    "cd_lpips": "CD-LPIPS", "ct": "CT",
}

DEFAULT_CLASS_LABEL = 0
SIGMA_DATA = 0.5


class _MinimalSchedulerConfig:
    sigma_data = SIGMA_DATA
    sigma_min = 0.002


class _MinimalScheduler:
    """Mock scheduler bypassing NumPy 2.x CMStochasticIterativeScheduler crash."""
    config = _MinimalSchedulerConfig()


def _load_unet(model_id: str, device: str, torch_dtype=None) -> dict:
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    unet = UNet2DModel.from_pretrained(model_id, subfolder="unet", **kwargs)
    unet = unet.to(device).eval()
    return {"pipe": None, "unet": unet, "scheduler": _MinimalScheduler()}


def _setup_for_jacobian(model_dict: dict) -> None:
    """Gradient checkpointing + disable flash/efficient SDPA for AD compatibility."""
    unet = model_dict.get("unet")
    if unet is None:
        return
    if hasattr(unet, "enable_gradient_checkpointing"):
        try:
            unet.enable_gradient_checkpointing()
        except Exception:
            pass
    try:
        import torch.backends.cuda as _cb
        _cb.enable_flash_sdp(False)
        _cb.enable_mem_efficient_sdp(False)
    except Exception:
        pass


def _load_by_name(
    name: str, device: str = "cuda", class_label: int = DEFAULT_CLASS_LABEL,
    torch_dtype=None,
) -> dict:
    model_id = MODEL_IDS[name]
    info = _load_unet(model_id, device, torch_dtype=torch_dtype)
    model_dict = {
        "name": name, "model_id": model_id,
        "pipe": info["pipe"], "unet": info["unet"], "scheduler": info["scheduler"],
        "label": MODEL_LABELS[name], "color": MODEL_COLORS[name],
        "class_label": class_label,
    }
    _setup_for_jacobian(model_dict)
    return model_dict


def load_edm_teacher(
    device: str = "cuda", class_label: int = DEFAULT_CLASS_LABEL,
    torch_dtype=torch.float16,
) -> dict:
    """Load the EDM teacher (defaults to float16)."""
    return _load_by_name("teacher", device, class_label, torch_dtype)


def load_cd_student(
    variant: Literal["cd_l2", "cd_lpips", "ct"],
    device: str = "cuda", class_label: int = DEFAULT_CLASS_LABEL,
    torch_dtype=torch.float16,
) -> dict:
    """Load one of the three CD/CT students."""
    if variant not in {"cd_l2", "cd_lpips", "ct"}:
        raise ValueError(f"Unknown student variant: {variant!r}")
    return _load_by_name(variant, device, class_label, torch_dtype)


def _class_labels_kwarg(model_dict: dict, device: str) -> dict:
    unet = model_dict["unet"]
    num_classes = getattr(unet.config, "num_class_embeds", None)
    if num_classes is not None and num_classes > 0:
        label = model_dict.get("class_label", DEFAULT_CLASS_LABEL)
        return {"class_labels": torch.tensor([label], device=device, dtype=torch.long)}
    return {}


def _sigma_to_t(scheduler, sigma: float) -> float:
    if hasattr(scheduler, "sigma_to_t"):
        return float(scheduler.sigma_to_t(sigma))
    return 1000.0 * 0.25 * math.log(sigma + 1e-44)


def denoise(
    model_dict: dict, noisy_image: torch.Tensor, sigma: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Differentiable denoiser at explicit sigma. Returns float32 clean estimate."""
    unet = model_dict["unet"]
    scheduler = model_dict["scheduler"]
    model_dtype = next(unet.parameters()).dtype
    noisy_image = noisy_image.to(device=device, dtype=torch.float32)

    sigma_data = getattr(scheduler.config, "sigma_data", SIGMA_DATA)
    sigma_min = getattr(scheduler.config, "sigma_min", 0.002)
    sigma_t = float(sigma)

    c_skip = sigma_data**2 / ((sigma_t - sigma_min)**2 + sigma_data**2)
    c_out = (sigma_t - sigma_min) * sigma_data / (sigma_t**2 + sigma_data**2)**0.5
    c_in = 1.0 / (sigma_t**2 + sigma_data**2)**0.5

    t = torch.tensor([_sigma_to_t(scheduler, sigma_t)], dtype=torch.float32).to(device)
    scaled_input = (noisy_image * c_in).to(model_dtype)
    t_unet = t.to(model_dtype)
    kwargs = _class_labels_kwarg(model_dict, device)

    with torch.enable_grad():
        model_output = unet(scaled_input, t_unet, return_dict=False, **kwargs)[0]

    return c_skip * noisy_image + c_out * model_output.to(torch.float32)


def denoise_via_pipeline(
    model_dict: dict, noisy_image: torch.Tensor, sigma: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Non-differentiable reference denoiser for sanity checks."""
    unet = model_dict["unet"]
    scheduler = model_dict["scheduler"]
    model_dtype = next(unet.parameters()).dtype
    noisy_image = noisy_image.to(device=device, dtype=torch.float32)

    sigma_data = getattr(scheduler.config, "sigma_data", SIGMA_DATA)
    sigma_min = getattr(scheduler.config, "sigma_min", 0.002)
    sigma_t = float(sigma)

    c_skip = sigma_data**2 / ((sigma_t - sigma_min)**2 + sigma_data**2)
    c_out = (sigma_t - sigma_min) * sigma_data / (sigma_t**2 + sigma_data**2)**0.5
    c_in = 1.0 / (sigma_t**2 + sigma_data**2)**0.5

    t = torch.tensor([_sigma_to_t(scheduler, sigma_t)], dtype=torch.float32).to(device)
    scaled_input = (noisy_image * c_in).to(model_dtype)
    t_unet = t.to(model_dtype)
    kwargs = _class_labels_kwarg(model_dict, device)

    with torch.no_grad():
        model_output = unet(scaled_input, t_unet, return_dict=False, **kwargs)[0]

    return c_skip * noisy_image + c_out * model_output.to(torch.float32)


def make_denoise_fn(
    model_dict: dict, sigma: float, device: str = "cuda",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Single-arg closure for halko_sym_eig."""
    def _fn(y: torch.Tensor) -> torch.Tensor:
        return denoise(model_dict, y, sigma, device)
    return _fn
