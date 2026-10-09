"""Gate for regenerated OOF prediction files.

Walks ``outputs/exp*_predictions/*.json``, tolerates trailing NUL padding from
transfers, understands both prediction schemas (the minimal PredictionLogger
one and exp7/exp15's rich one), and asserts for every file:
  1. no duplicate pid within a fold,
  2. no pid shared across folds (the leakage bug this whole change fixes),
  3. exactly ``N_FOLDS`` folds,
  4. (optional) the unique-pid count matches an expected-cohort map.

With ``--expect MANIFEST`` it also fails when a required file is missing: each
non-blank, non-# line of the manifest is a glob relative to the root that must
match at least one prediction file. The rerun scripts write this manifest, so a
config that crashed (and so wrote nothing) cannot slip through the gate.

Run as ``python -m shared.verify_oof outputs [--expect manifest.txt]``
(exit non-zero on any violation).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from shared.cohort import assert_oof_no_leakage

# (regex on the "<dir>/<file>" path, expected unique-pid count), most specific
# first - the first match wins. Cohort size is set by the modality combination;
# the EEG sizes are those of the version-2 cache built on M3 on 2026-10-09
# (analysis plan Addendum C): text+eeg = 108, + one text = 117, + one eeg = 148,
# clinical(+smiles) = 198. Every EEG feature set (REVE, LaBraM) is written from
# that cache, so exp15 has the same 108 as exp7.
# Anchored on '_' delimiters so 'exp1' cannot match an 'exp11' filename.
EXPECTED_COUNTS = [
    (r"exp7_predictions/predictions_oof", 108),        # quad headline (7a) + 7b MoE
    (r"exp15_predictions/predictions_oof", 108),       # quad on stored EEG features (both feature sets)
    (r"exp16_predictions/predictions_oof", 108),       # reduced-capacity quad (same 108 as exp7)
    (r"exp17_predictions/predictions_oof", 83),        # focal-only quad (focal subset of the 108)
    (r"exp9_predictions/predictions_oof", 148),        # standalone EEG encoder sweep
    (r"exp19_predictions/predictions_oof_exp19_(D|D-tok|D-split|T6a)_", 117),  # configurations needing an EEG report
    (r"exp19_predictions/predictions_oof_exp19_", 198),          # serialised/tabular clinical cohort
    (r"exp11_6b", 148), (r"exp11_(3a|7a)", 108),       # exp11 sub-configs (base leads the name)
    (r"_exp3[ab]_", 108),
    (r"_exp1[ab]_", 117), (r"_exp5b_", 117), (r"_exp6a_", 117),
    (r"_exp2_", 148), (r"_exp5c_", 148), (r"_exp6b_", 148),
    (r"_exp4[ab]_", 198), (r"_exp5a_", 198),
]


N_FOLDS = 5


def load_json_nul_tolerant(path: Path) -> dict:
    raw = path.read_bytes().replace(b"\x00", b"").strip()
    return json.loads(raw)


def fold_pid_lists(payload: dict) -> list[list[str]]:
    """Per-fold pid lists, for both schemas (both nest pids under 'folds')."""
    return [[str(p) for p in fold["pids"]] for fold in payload.get("folds", [])]


def expected_for(name: str) -> int | None:
    for pattern, n in EXPECTED_COUNTS:
        if re.search(pattern, name):
            return n
    return None


def verify_file(path: Path) -> list[str]:
    """Return a list of problems for one file (empty == passed)."""
    problems: list[str] = []
    try:
        payload = load_json_nul_tolerant(path)
        folds = fold_pid_lists(payload)  # a non-dict/malformed payload raises here
    except Exception as exc:  # noqa: BLE001 - report per-file, never abort the run
        return [f"unreadable/malformed: {exc}"]
    if not folds:
        return ["no folds / pids"]
    if len(folds) != N_FOLDS:
        problems.append(f"{len(folds)} folds != expected {N_FOLDS}")
    try:
        assert_oof_no_leakage(folds)
    except AssertionError as exc:
        problems.append(str(exc))
    unique = {p for fold in folds for p in fold}
    rel = f"{path.parent.name}/{path.name}"
    want = expected_for(rel)
    if want is not None and len(unique) != want:
        problems.append(f"unique pids {len(unique)} != expected {want}")
    return problems


def missing_expected(root: Path, manifest: Path) -> list[str]:
    """Manifest globs (relative to root) that match no file."""
    missing = []
    for line in manifest.read_text().splitlines():
        pattern = line.strip()
        if pattern and not pattern.startswith("#") and not any(not is_smoke_file(p) for p in root.glob(pattern)):
            missing.append(pattern)
    return missing


def is_smoke_file(path: Path) -> bool:
    """A dry-run file written by ``rerun_clean.sh smoke`` (``--smoke``), never a result."""
    return "_smoke" in path.stem


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default="outputs")
    parser.add_argument("--expect", type=Path, default=None,
                        help="manifest of required prediction-file globs")
    args = parser.parse_args(argv[1:])
    root = Path(args.root)
    # OOF files only: in-sample predictions have no folds and legitimately
    # repeat pids, so the leakage/cohort checks don't apply to them.
    files = sorted(root.glob("exp*_predictions/predictions_oof*.json")) + sorted(
        root.glob("exp4_decomposition/predictions_oof*.json"))
    smoke_files = [f for f in files if is_smoke_file(f)]
    files = [f for f in files if not is_smoke_file(f)]
    if smoke_files:
        print(f"note: {len(smoke_files)} smoke-run file(s) ignored (rerun_clean.sh smoke relocates them)", file=sys.stderr)
    if not files:
        print(f"no prediction files under {root}", file=sys.stderr)
        return 1
    missing = missing_expected(root, args.expect) if args.expect else []
    for pattern in missing:
        print(f"FAIL missing expected file: {pattern}", file=sys.stderr)
    failed = 0
    for path in files:
        problems = verify_file(path)
        rel = f"{path.parent.name}/{path.name}"
        if problems:
            failed += 1
            for p in problems:
                print(f"FAIL {rel}: {p}", file=sys.stderr)
        else:
            print(f"ok   {rel}")
    print(f"\n{len(files) - failed}/{len(files)} files passed")
    if args.expect:
        print(f"{len(missing)} expected file pattern(s) missing")
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
