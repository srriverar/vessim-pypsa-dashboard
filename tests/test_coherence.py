"""The audit in tools/check_coherence.py, run on a sample inside pytest.

The script is the full thing (every step, every case, written to
``reports/coherence.md``); this test makes sure it keeps working and that no
case regresses to an incoherent trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from check_coherence import audit_case, audit_papers  # noqa: E402
from vessim_pypsa_dashboard.vessim_runner import CASE_ORDER  # noqa: E402


@pytest.mark.parametrize("key", CASE_ORDER)
def test_case_is_coherent(key):
    audit = audit_case(key, samples=2)
    failures = [(result.check, result.worst, result.tolerance) for result in audit.failures]
    assert not failures, failures
    assert len(audit.results) >= 18


def test_paper_reproductions_are_coherent():
    audit = audit_papers()
    assert not audit.failures, [(r.check, r.worst) for r in audit.failures]
