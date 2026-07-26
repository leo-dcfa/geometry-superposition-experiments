"""Tests for the Phase-00 gate criteria themselves.

The gates decide whether Phase 1 opens, so the arithmetic that renders a
verdict is worth testing as much as the geometry is. These pin the two
criteria that were revised during 00c: the one-unit granularity allowance, and
the meaning of the sub-gate's verdict.
"""

from __future__ import annotations

from experiments.csc1.phase00.run_00c import PARITY_TOLERANCE, parity_table


def _cell(arm_label, head, n_features, dead_init, dead_trained, recovery, **kw):
    return {
        "arm_label": arm_label,
        "head": head,
        "weight_decay": 0.0,
        "dim": 8,
        "n_features": n_features,
        "capacity_binding": False,
        "dead_at_init": dead_init,
        "dead_after_training": dead_trained,
        "recovery_rate": recovery,
        **kw,
    }


def _pair(head, n_features, curved_dead, curved_recovery=1.0):
    """A Euclidean baseline plus one curved arm differing only in dead units."""
    return [
        _cell("euclidean", head, n_features, 0.0, 0.0, 1.0),
        _cell("curved(K=-1)", head, n_features, 0.0, curved_dead, curved_recovery),
    ]


def test_single_dead_unit_does_not_fail_parity_at_small_n():
    """At N=3 one dead unit is a 33% gap by construction, not unfairness.

    The dead-unit fraction is quantized in steps of 1/N, so a flat 5% criterion
    is finer than the metric can express. This was a real false positive in the
    first gated 00c run.
    """
    table = parity_table(_pair("softmax", n_features=3, curved_dead=1 / 3))
    assert table["n_failures"] == 0
    assert table["verdict"] == "PASS"


def test_two_dead_units_still_fails_at_small_n():
    """The allowance is one unit, not a blanket exemption for small N."""
    table = parity_table(_pair("softmax", n_features=3, curved_dead=2 / 3))
    assert table["n_failures"] == 1


def test_real_unfairness_fails_at_large_n():
    table = parity_table(_pair("softmax", n_features=64, curved_dead=0.20))
    assert table["n_failures"] == 1
    assert table["verdict"] == "FAIL"
    assert table["per_head_verdict"]["softmax"] == "DISQUALIFIED"


def test_tolerance_is_never_tighter_than_one_unit():
    table = parity_table(_pair("softmax", n_features=8, curved_dead=1 / 8))
    assert table["failures"] == []
    # and at N large enough, the flat tolerance takes over
    table = parity_table(_pair("softmax", n_features=100, curved_dead=0.06))
    assert table["failures"][0]["effective_tolerance"] == PARITY_TOLERANCE


def test_parity_between_two_equally_dead_arms_is_not_a_pass():
    """Agreement between two broken readouts is not fairness."""
    cells = [
        _cell("euclidean", "rbf", 16, 0.0, 1.0, 0.0),
        _cell("curved(K=-1)", "rbf", 16, 0.0, 1.0, 0.0),
    ]
    table = parity_table(cells)
    assert table["n_failures"] == 0, "the two arms do agree..."
    assert table["verdict"] == "FAIL", "...but neither recovers anything"
    assert table["per_head_verdict"]["rbf"] == "DISQUALIFIED"


def test_gate_passes_when_one_head_works_even_if_others_fail():
    """00c's job is to find a usable instrument, not to bless every candidate."""
    cells = _pair("softmax", 16, curved_dead=0.0) + _pair("rbf", 16, curved_dead=0.9, curved_recovery=0.0)
    table = parity_table(cells)
    assert table["verdict"] == "PASS"
    assert table["per_head_verdict"] == {"rbf": "DISQUALIFIED", "softmax": "USABLE"}


def test_crowded_cells_are_excluded_from_the_gate():
    """On crowded shapes a dead unit is correct behaviour, not unfairness."""
    crowded = [
        _cell("euclidean", "softmax", 64, 0.0, 0.10, 1.0, capacity_binding=True),
        _cell("curved(K=-1)", "softmax", 64, 0.0, 0.90, 1.0, capacity_binding=True),
    ]
    for c in crowded:
        c["capacity_binding"] = True
    clean = _pair("softmax", 16, curved_dead=0.0)
    assert parity_table(clean + crowded, gated_only=True)["n_failures"] == 0
    assert parity_table(clean + crowded, gated_only=False)["n_failures"] == 1
