"""Repository and data-directory paths shared by every pipeline.

Kept free of torch, MNE and the experiment packages so the cache builder, the
cohort helpers and CPU-only jobs can import it without loading a deep-learning
stack. Data never lives in this repo: the CSVs resolve through ``asm_data``
(``$ASM_DATA_DIR`` or the repo sibling ``../asm_data``).
"""

from __future__ import annotations

import os
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent


def find_asm_data_dir() -> Path:
    """``$ASM_DATA_DIR`` if set, else ``asm_data`` beside the experiments repo."""
    env = os.environ.get("ASM_DATA_DIR")
    if env and Path(env).exists():
        return Path(env)
    sibling = EXPERIMENTS_ROOT.parent / "asm_data"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "asm_data not found: set ASM_DATA_DIR or place asm_data beside the experiments repo."
    )


_ASM_DATA_DIR = find_asm_data_dir()
ALFRED_CSV = _ASM_DATA_DIR / "alfred_1st_regimen.csv"
HEP_CSV = _ASM_DATA_DIR / "hep_1st_regimen.csv"
