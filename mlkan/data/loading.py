from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_dataset(path, splits, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False))}
    for split, values in splits.items():
        for key, value in values.items():
            arrays[f"{split}_{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_dataset(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        splits = {}
        for split in ("train", "val", "test"):
            prefix = f"{split}_"
            values = {
                key[len(prefix):]: np.asarray(archive[key])
                for key in archive.files if key.startswith(prefix)
            }
            if values:
                splits[split] = values
    return splits, metadata


def validate_dataset(splits, metadata):
    errors = []
    for split, values in splits.items():
        if "x" not in values or "y" not in values:
            errors.append(f"{split}: missing x or y")
            continue
        if len(values["x"]) != len(values["y"]):
            errors.append(f"{split}: x/y sample count mismatch")
        if not np.all(np.isfinite(values["x"])):
            errors.append(f"{split}: x contains non-finite values")
        if not np.all(np.isfinite(values["y"])):
            errors.append(f"{split}: y contains non-finite values")
        if "group_id" in values:
            for other, other_values in splits.items():
                if other <= split or "group_id" not in other_values:
                    continue
                overlap = np.intersect1d(
                    np.unique(values["group_id"]),
                    np.unique(other_values["group_id"]),
                )
                if len(overlap):
                    errors.append(f"group leakage between {split} and {other}: {overlap[:5]}")
    if metadata.get("dimension") not in (1, 2):
        errors.append("metadata.dimension must be 1 or 2")
    return errors
