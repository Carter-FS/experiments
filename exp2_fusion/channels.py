"""Channel naming for the EEG cache: the 19 standard 10-20 scalp channels and the
patient-id rule for EDF file names. Kept free of MNE and torch so configs can import it."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Standard 10-20 modern-naming canonical 19 scalp EEG channels. Both Alfred and HEP
# recordings include these (Alfred natively, HEP under the old MCN names T3/T4/T5/T6).
# The non-EEG channels present in the EDFs but typed as 'eeg' (EMG+/-, PG1/2, A1/A2,
# ECG+/-, ekg, sop) are dropped by the name filter.
STD_19_TEN_TWENTY = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8",
    "O1", "O2",
)
N_CHANNELS = len(STD_19_TEN_TWENTY)

# Old MCN names -> modern. Applied case-insensitively to all incoming channel names
# before the 19-channel intersection check.
OLD_TO_MODERN_NAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalise_channel_name(name: str) -> str:
    """Return the canonical modern uppercase form of an EEG channel name."""
    n = name.strip().upper()
    return OLD_TO_MODERN_NAME.get(n, n)


def extract_patient_id(filename: str) -> Optional[str]:
    """Patient id from an EEG file name: the part before the first comma, else before
    the first underscore (``083_7085712_15-2-2019.edf`` -> ``083``;
    ``093,EEG,07022018.edf`` -> ``093``); ``None`` when the name has neither."""
    basename = Path(filename).stem
    if "," in basename:
        return basename.split(",")[0].strip()
    if "_" in basename:
        return basename.split("_")[0].strip()
    return None
