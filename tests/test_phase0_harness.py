"""Tests for the Fase 0 evaluation harness (metrics + aggregation, no API)."""

import pytest

from grubrics_science.phase0.harness import (
    aggregate_generator,
    compute_metrics,
)


def _answers(sources):
    """Build minimal answer dicts from a list of (source, hack_family)."""
    out = []
    for src, fam in sources:
        out.append({"text": "x", "source": src, "hack_family": fam})
    return out


class TestComputeMetrics:
    def test_perfect_alignment_and_hack_separation(self):
        # 2 honest (high anchor), 2 hacks (low anchor); rubric agrees perfectly
        answers = _answers([
            ("honest", None), ("honest", None),
            ("hack", "keyword_stuffing"), ("hack", "completeness_filler"),
        ])
        anchor = [8.0, 6.0, 2.0, 1.0]
        rubric_scores = [1.0, 0.8, 0.2, 0.1]
        m = compute_metrics(rubric_scores, answers, anchor)
        assert m["alignment"] == pytest.approx(1.0)
        assert m["hack_gap"] == pytest.approx(0.9 - 0.15)
        assert m["hack_detection"] == pytest.approx(1.0)  # both hacks below honest median

    def test_rubric_fooled_by_hacks(self):
        # rubric gives hacks HIGH scores -> negative hack_gap, low detection
        answers = _answers([
            ("honest", None), ("honest", None),
            ("hack", "keyword_stuffing"),
        ])
        anchor = [8.0, 6.0, 1.0]
        rubric_scores = [0.3, 0.2, 0.9]  # hack scored highest
        m = compute_metrics(rubric_scores, answers, anchor)
        assert m["hack_gap"] < 0
        assert m["hack_detection"] == 0.0

    def test_family_means(self):
        answers = _answers([
            ("honest", None),
            ("hack", "keyword_stuffing"),
            ("hack", "keyword_stuffing"),
            ("hack", "partial_compound"),
        ])
        anchor = [8.0, 1.0, 2.0, 3.0]
        rubric_scores = [1.0, 0.1, 0.3, 0.5]
        m = compute_metrics(rubric_scores, answers, anchor)
        assert m["family_means"]["keyword_stuffing"] == pytest.approx(0.2)
        assert m["family_means"]["partial_compound"] == pytest.approx(0.5)

    def test_no_hacks_gives_none(self):
        answers = _answers([("honest", None), ("honest", None)])
        m = compute_metrics([1.0, 0.5], answers, [5.0, 3.0])
        assert m["hack_gap"] is None
        assert m["hack_detection"] is None


class TestAggregateGenerator:
    def test_aggregate_basic(self):
        results = [
            {"parse_ok": True, "n_criteria": 5,
             "metrics": {"alignment": 0.8, "hack_gap": 0.5, "hack_detection": 1.0}},
            {"parse_ok": True, "n_criteria": 7,
             "metrics": {"alignment": 0.6, "hack_gap": 0.3, "hack_detection": 0.5}},
            {"parse_ok": False, "n_criteria": 0,
             "metrics": {"alignment": None, "hack_gap": None, "hack_detection": None}},
        ]
        agg = aggregate_generator(results)
        assert agg["alignment"]["mean"] == pytest.approx(0.7)
        assert agg["alignment"]["n"] == 2
        assert agg["parse_ok_rate"] == pytest.approx(0.667, abs=1e-3)
        assert agg["mean_n_criteria"] == pytest.approx(6.0)
        assert agg["n_questions"] == 3

    def test_all_parse_fail(self):
        results = [
            {"parse_ok": False, "n_criteria": 0,
             "metrics": {"alignment": None, "hack_gap": None, "hack_detection": None}},
        ]
        agg = aggregate_generator(results)
        assert agg["alignment"] is None
        assert agg["parse_ok_rate"] == 0.0
