from __future__ import annotations

from .fno import build_fno
from .rescnn import build_rescnn
from .unet import build_unet


MODEL_SPECS = {
    "M0": {"label": "U-Net", "family": "unet", "correction": "none", "gate": "fixed"},
    "M1": {"label": "U-Net + single KAN-Middle", "family": "unet", "correction": "single", "gate": "fixed"},
    "M2": {"label": "U-Net + three-level KAN, fixed", "family": "unet", "correction": "kan", "gate": "fixed"},
    "M3": {"label": "U-Net + three-level KAN, spatial", "family": "unet", "correction": "kan", "gate": "spatial"},
    "M4": {"label": "U-Net + three-level KAN, SE", "family": "unet", "correction": "kan", "gate": "se"},
    "M5": {"label": "ML-KAN-U-Net, joint gate", "family": "unet", "correction": "kan", "gate": "joint"},
    "M6": {"label": "U-Net + parameter-matched convolution", "family": "unet", "correction": "conv", "gate": "fixed"},
    "M7": {"label": "U-Net + parameter-matched MLP", "family": "unet", "correction": "mlp", "gate": "fixed"},
    "M8": {"label": "ResCNN", "family": "rescnn", "correction": "none", "gate": "fixed"},
    "M9": {"label": "ResCNN + multi-stage KAN", "family": "rescnn", "correction": "multi", "gate": "fixed"},
    "M10": {"label": "FNO", "family": "fno", "physics": False},
    "M11": {"label": "FNO + discrete physics", "family": "fno", "physics": True},
}



def _normalize_correction_levels(levels):
    """
    Normalize correction level specifications to integer encoder levels.

    Canonical mapping:
        shallow -> 0
        middle  -> 1
        deep    -> 2

    Integer specifications 0/1/2 are preserved.

    This normalization prevents a silent graph-disabling bug caused by
    comparing integer encoder levels against string level names.
    """
    if levels is None:
        return (0, 1, 2)

    if isinstance(levels, (str, int)):
        levels = [levels]

    mapping = {
        "shallow": 0,
        "middle": 1,
        "deep": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }

    normalized = []

    for level in levels:
        if isinstance(level, str):
            key = level.strip().lower()

            if key not in mapping:
                raise ValueError(
                    f"Unknown correction level: {level!r}. "
                    "Expected shallow/middle/deep or 0/1/2."
                )

            value = mapping[key]

        elif isinstance(level, int):
            value = int(level)

        else:
            try:
                value = int(level)
            except Exception as exc:
                raise ValueError(
                    f"Unsupported correction level: {level!r}"
                ) from exc

        if value not in (0, 1, 2):
            raise ValueError(
                f"Correction level must be 0, 1 or 2, got {value}"
            )

        if value not in normalized:
            normalized.append(value)

    return tuple(normalized)


def build_model(model_id, input_shape, dimension, config=None):
    config = dict(config or {})
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model id: {model_id}")
    spec = MODEL_SPECS[model_id]
    family = spec["family"]
    if family == "unet":
        return build_unet(
            input_shape=input_shape,
            dimension=dimension,
            base_channels=config.get("base_channels", 32),
            depth=config.get("depth", 3),
            kernel_size=config.get("kernel_size", 5),
            activation=config.get("activation", "gelu"),
            correction=config.get("correction_override", spec["correction"]),
            correction_levels=_normalize_correction_levels(
            config.get("correction_levels", [0, 1, 2])
        ),
            gammas=tuple(config.get("gammas", [0.02, 0.05, 0.08])),
            kan_widths=tuple(config.get("kan_widths", [8, 16, 32])),
            kan_grid_size=config.get("kan_grid_size", 8),
            kan_basis=config.get("kan_basis", "bspline"),
            spline_order=config.get("spline_order", 3),
            share_kan=config.get("share_kan", False),
            gate=config.get("gate_override", spec.get("gate", "fixed")),
            correction_regularization=config.get("correction_regularization", 0.0),
        )
    if family == "rescnn":
        if dimension != 1:
            raise ValueError("M8/M9 ResCNN cross-backbone models are defined for 1D tasks")
        return build_rescnn(
            input_shape=input_shape,
            channels=config.get("rescnn_channels", 64),
            blocks=config.get("rescnn_blocks", 6),
            correction=spec["correction"],
            gammas=tuple(config.get("gammas", [0.02, 0.05, 0.08])),
            widths=tuple(config.get("rescnn_kan_widths", [16, 16, 16])),
            grid_size=config.get("kan_grid_size", 8),
            kan_basis=config.get("kan_basis", "bspline"),
            spline_order=config.get("spline_order", 3),
            gate=config.get("cross_backbone_gate", "fixed"),
            correction_regularization=config.get("correction_regularization", 0.0),
        )
    if family == "fno":
        return build_fno(
            input_shape=input_shape, dimension=dimension,
            width=config.get("fno_width", 48), modes=config.get("fno_modes", 16),
            blocks=config.get("fno_blocks", 4),
        )
    raise ValueError(family)
