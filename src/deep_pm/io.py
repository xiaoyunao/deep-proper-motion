from hashlib import sha256
import json

import numpy as np


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.with_suffix(".part")
    draft.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False) + "\n")
    draft.replace(path)


def digest(path):
    with path.open("rb") as source:
        state = sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()
