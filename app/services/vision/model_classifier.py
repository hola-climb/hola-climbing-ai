"""Optional PyTorch GRU classifier for pose dynamic/static prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without ml deps
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch() -> Any:
    """Return torch module or raise a clear install hint."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for learned pose classification. "
            "Install it with `uv sync --group ml`."
        ) from _TORCH_IMPORT_ERROR
    return torch


if nn is not None:

    class PoseSequenceClassifier(nn.Module):
        """Small GRU binary classifier. Output logit > 0 means dynamic."""

        def __init__(self, input_size: int = 132, hidden_size: int = 64, num_layers: int = 1) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> Any:
            _, hidden = self.gru(x)
            last_hidden = hidden[-1]
            return self.head(last_hidden).squeeze(-1)

else:  # pragma: no cover

    class PoseSequenceClassifier:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            require_torch()


@dataclass(frozen=True)
class LoadedPoseClassifier:
    model: PoseSequenceClassifier
    target_frames: int
    hidden_size: int
    num_layers: int
    input_size: int
    feature_set: str
    metrics: dict[str, float]


def save_checkpoint(
    path: Path,
    *,
    model: PoseSequenceClassifier,
    target_frames: int,
    hidden_size: int,
    num_layers: int,
    metrics: dict[str, float],
    input_size: int = 132,
    feature_set: str = "raw",
) -> None:
    """Save model state and metadata."""
    torch_module = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "model_state": model.state_dict(),
            "target_frames": target_frames,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "input_size": input_size,
            "feature_set": feature_set,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path) -> LoadedPoseClassifier:
    """Load a saved pose classifier checkpoint."""
    torch_module = require_torch()
    payload = torch_module.load(path, map_location="cpu")
    hidden_size = int(payload["hidden_size"])
    num_layers = int(payload["num_layers"])
    input_size = int(payload.get("input_size", 132))
    model = PoseSequenceClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return LoadedPoseClassifier(
        model=model,
        target_frames=int(payload["target_frames"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        input_size=input_size,
        feature_set=str(payload.get("feature_set", "raw")),
        metrics=cast(dict[str, float], payload.get("metrics", {})),
    )


def predict_dynamic_probability(
    model: PoseSequenceClassifier,
    x: NDArray[np.float32],
) -> float:
    """Predict dynamic probability for one flattened pose sequence `(T, 132)`."""
    torch_module = require_torch()
    input_size = int(model.gru.input_size)
    if x.ndim != 2 or x.shape[1] != input_size:
        raise ValueError(f"x must have shape (T, {input_size}), got {x.shape}")
    model.eval()
    with torch_module.no_grad():
        tensor = torch_module.from_numpy(x.astype(np.float32)).unsqueeze(0)
        logit = model(tensor)
        prob = torch_module.sigmoid(logit)[0].item()
    return float(prob)
