"""Tests for curriculum scheduler and data preparation with cache.

Run with: pytest tests/test_curriculum.py -v
No GPU required. No API calls.
"""

import json
import pytest


# =========================================================================
# CurriculumScheduler tests
# =========================================================================

class TestCurriculumScheduler:
    """Test CurriculumScheduler phase tracking and step boundaries."""

    def test_default_phases(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=2000)

        assert len(scheduler.phases) == 3
        # Phase 1: 80% verif, 20% open, 40% of steps = 800 steps
        assert scheduler.phases[0].verif_ratio == 0.8
        assert scheduler.phases[0].open_ratio == 0.2
        # Phase 2: 50/50, 30% = 600 steps
        assert scheduler.phases[1].verif_ratio == 0.5
        # Phase 3: 20/80, 30% = 600 steps
        assert scheduler.phases[2].verif_ratio == 0.2

    def test_step_boundaries(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=1000)
        boundaries = scheduler.get_phase_boundaries()

        # 40% + 30% + 30% = 400 + 300 + 300
        assert boundaries == [400, 700, 1000]

    def test_get_phase_index(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=1000)

        assert scheduler.get_phase_index(0) == 0
        assert scheduler.get_phase_index(399) == 0
        assert scheduler.get_phase_index(400) == 1
        assert scheduler.get_phase_index(699) == 1
        assert scheduler.get_phase_index(700) == 2
        assert scheduler.get_phase_index(999) == 2

    def test_get_data_file(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=1000, data_dir="/tmp/data")

        assert "phase1" in scheduler.get_data_file(0)
        assert "phase2" in scheduler.get_data_file(500)
        assert "phase3" in scheduler.get_data_file(800)

    def test_needs_data_switch(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=1000)

        assert not scheduler.needs_data_switch(10, 11)
        assert not scheduler.needs_data_switch(398, 399)
        assert scheduler.needs_data_switch(399, 400)
        assert scheduler.needs_data_switch(699, 700)

    def test_custom_phases(self):
        from grubrics_science.training.curriculum import (
            CurriculumPhase,
            CurriculumScheduler,
        )

        phases = [
            CurriculumPhase(verif_ratio=1.0, open_ratio=0.0, fraction=0.5),
            CurriculumPhase(verif_ratio=0.0, open_ratio=1.0, fraction=0.5),
        ]
        scheduler = CurriculumScheduler(total_steps=100, phases=phases)

        assert scheduler.get_phase_index(0) == 0
        assert scheduler.get_phase_index(49) == 0
        assert scheduler.get_phase_index(50) == 1
        assert scheduler.get_phase_index(99) == 1

    def test_lr_scale(self):
        from grubrics_science.training.curriculum import (
            CurriculumPhase,
            CurriculumScheduler,
        )

        phases = [
            CurriculumPhase(verif_ratio=0.8, open_ratio=0.2, fraction=0.5, lr_scale=1.0),
            CurriculumPhase(verif_ratio=0.2, open_ratio=0.8, fraction=0.5, lr_scale=0.5),
        ]
        scheduler = CurriculumScheduler(total_steps=100, phases=phases)

        assert scheduler.get_lr_scale(10) == 1.0
        assert scheduler.get_lr_scale(60) == 0.5

    def test_summary(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=1000)
        summary = scheduler.summary()

        assert "Phase 1" in summary
        assert "Phase 2" in summary
        assert "Phase 3" in summary
        assert "1000 total steps" in summary

    def test_fraction_normalization(self):
        from grubrics_science.training.curriculum import (
            CurriculumPhase,
            CurriculumScheduler,
        )

        # Fractions don't sum to 1
        phases = [
            CurriculumPhase(verif_ratio=0.8, open_ratio=0.2, fraction=4),
            CurriculumPhase(verif_ratio=0.2, open_ratio=0.8, fraction=6),
        ]
        scheduler = CurriculumScheduler(total_steps=100, phases=phases)

        boundaries = scheduler.get_phase_boundaries()
        assert boundaries == [40, 100]

    def test_get_ratios(self):
        from grubrics_science.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(total_steps=100)
        ratios = scheduler.get_ratios()

        assert ratios == [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]


# =========================================================================
# prepare_mixed_with_cache tests
# =========================================================================

class TestPrepareMixedWithCache:
    """Test prepare_mixed_with_cache passes cache to adapters."""

    @pytest.fixture
    def gsm8k_cache(self, tmp_path):
        cache = tmp_path / "gsm8k_cache.jsonl"
        # We need entries that match the actual GSM8K dataset questions
        # Since we can't predict exact questions, just create the file
        cache.write_text("")
        return str(cache)

    @pytest.fixture
    def fs_dataset(self, tmp_path):
        """Create a minimal FrontierScience dataset."""
        ds = tmp_path / "test.jsonl"
        record = {
            "problem": "Derive X.",
            "answer": "Points: 5.0, Item: Correct\nPoints: 5.0, Item: Units",
            "subject": "physics",
        }
        ds.write_text(json.dumps(record) + "\n")
        return str(ds)

    @pytest.fixture
    def fs_cache(self, tmp_path):
        cache = tmp_path / "fs_cache.jsonl"
        entry = {
            "question_id": "0",
            "question": "Derive X.",
            "golden_rubric": "Points: 5.0, Item: Correct\nPoints: 5.0, Item: Units",
            "subject": "physics",
            "answers": ["Good answer", "Bad answer"],
            "gold_scores": [0.9, 0.3],
        }
        cache.write_text(json.dumps(entry) + "\n")
        return str(cache)


# =========================================================================
# parse_phases tests
# =========================================================================

class TestParsePhases:
    """Test phase string parsing."""

    def test_parse_three_values(self):
        from grubrics_science.training.curriculum import parse_phases

        phases = parse_phases(["0.8:0.2:0.4", "0.5:0.5:0.3", "0.2:0.8:0.3"])

        assert len(phases) == 3
        assert phases[0].verif_ratio == 0.8
        assert phases[0].open_ratio == 0.2
        assert phases[0].fraction == 0.4
        assert phases[0].lr_scale == 1.0  # default

    def test_parse_four_values_with_lr(self):
        from grubrics_science.training.curriculum import parse_phases

        phases = parse_phases(["1.0:0.0:0.5:1.0", "0.0:1.0:0.5:0.3"])

        assert phases[1].lr_scale == 0.3

    def test_parse_invalid_format(self):
        from grubrics_science.training.curriculum import parse_phases

        with pytest.raises(ValueError):
            parse_phases(["0.8:0.2"])  # only 2 values
