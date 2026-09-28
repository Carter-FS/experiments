"""Guard the Melbourne outcome polarity (analysis plan Addendum B.1).

The raw CSV codes 1 = success, 2 = failure. Only shared/cohort.py may map it;
a literal map anywhere else is how the inverted {1: 0, 2: 1} survived for
months, so any such literal fails this test.
"""
import re
from pathlib import Path

from shared.cohort import OUTCOME_MAPPING

ROOT = Path(__file__).resolve().parents[2]
LITERAL = re.compile(r"\{\s*1\s*:\s*[01]\s*,\s*2\s*:\s*[01]\s*\}")
SKIP_DIRS = {"outputs", ".venv-others", ".venv-moler", ".git", "node_modules", "_prepull_backup"}


def test_mapping_matches_data_dictionary():
    assert OUTCOME_MAPPING == {1: 1, 2: 0}


def test_no_literal_outcome_map_outside_shared_cohort():
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.startswith("_prepull_backup") for part in rel.parts):
            continue
        if rel.as_posix() in ("shared/cohort.py", "shared/tests/test_outcome_polarity.py"):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if LITERAL.search(line) and "outcome" in text.lower():
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "literal outcome maps outside shared/cohort.py:\n" + "\n".join(offenders)
