"""Tests for Fase 0 components: hacks generation + holistic panel (no API)."""

import asyncio

import pytest

from grubrics_science.phase0 import hacks, panel


# ---------------------------------------------------------------------------
# hacks
# ---------------------------------------------------------------------------

class TestHackFamilies:
    def test_four_families_present(self):
        assert set(hacks.DEFAULT_FAMILIES) == {
            "keyword_stuffing",
            "completeness_filler",
            "implicit_as_explicit",
            "partial_compound",
        }

    def test_families_have_instructions(self):
        for key, fam in hacks.HACK_FAMILIES.items():
            assert fam.key == key
            assert fam.instruction
            assert fam.name
            assert fam.description

    def test_prompt_includes_question_and_reference(self):
        fam = hacks.HACK_FAMILIES["keyword_stuffing"]
        prompt = hacks._build_hack_prompt("Q_MARKER", "REF_MARKER", fam)
        assert "Q_MARKER" in prompt
        assert "REF_MARKER" in prompt
        assert fam.instruction in prompt


class FakeGenClient:
    def __init__(self, reply="HACK TEXT"):
        self.reply = reply
        self.calls = 0

    async def generate(self, prompt, system_prompt=None, max_tokens=512,
                       temperature=1.0, **kwargs):
        self.calls += 1
        return self.reply


class TestGenerateHacks:
    def test_one_per_family(self):
        client = FakeGenClient()
        sem = asyncio.Semaphore(4)
        result = asyncio.run(hacks.generate_hacks_for_question(
            client, "question", "reference answer", sem,
        ))
        assert len(result) == 4
        assert {h["hack_family"] for h in result} == set(hacks.DEFAULT_FAMILIES)
        assert all(h["text"] == "HACK TEXT" for h in result)

    def test_subset_of_families(self):
        client = FakeGenClient()
        sem = asyncio.Semaphore(4)
        result = asyncio.run(hacks.generate_hacks_for_question(
            client, "q", "ref", sem, families=["keyword_stuffing"],
        ))
        assert len(result) == 1
        assert result[0]["hack_family"] == "keyword_stuffing"

    def test_empty_generation_dropped(self):
        client = FakeGenClient(reply="   ")
        sem = asyncio.Semaphore(4)
        result = asyncio.run(hacks.generate_hacks_for_question(
            client, "q", "ref", sem,
        ))
        assert result == []


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

class TestPanelParsing:
    def test_parse_scores_code_block(self):
        resp = '```json\n{"scores": {"0": 80, "1": 20, "2": 50}, "best": 0, "worst": 1}\n```'
        scores = panel._parse_panel_scores(resp, 3)
        assert scores == [80.0, 20.0, 50.0]

    def test_parse_scores_raw(self):
        resp = '{"scores": {"0": 10, "1": 90}}'
        assert panel._parse_panel_scores(resp, 2) == [10.0, 90.0]

    def test_parse_missing_index_fails(self):
        resp = '{"scores": {"0": 10}}'
        assert panel._parse_panel_scores(resp, 2) is None

    def test_parse_garbage_fails(self):
        assert panel._parse_panel_scores("no json here", 2) is None
        assert panel._parse_panel_scores("", 2) is None


class TestRanksFromScores:
    def test_basic_ordering(self):
        # scores: answer1 best (90), answer2 mid (50), answer0 worst (10)
        ranks = panel._ranks_from_scores([10.0, 90.0, 50.0])
        assert ranks[1] == 0.0   # best -> rank 0
        assert ranks[2] == 1.0
        assert ranks[0] == 2.0

    def test_ties_get_average_rank(self):
        ranks = panel._ranks_from_scores([50.0, 50.0, 10.0])
        # two tied at top -> ranks 0 and 1 averaged to 0.5
        assert ranks[0] == 0.5
        assert ranks[1] == 0.5
        assert ranks[2] == 2.0


class TestPairwiseAgreement:
    def test_identical_judges_perfect(self):
        a = [80.0, 50.0, 20.0]
        ag = panel._pairwise_agreement([a, list(a)])
        assert ag == pytest.approx(1.0)

    def test_single_judge_none(self):
        assert panel._pairwise_agreement([[1.0, 2.0]]) is None

    def test_opposite_judges_negative(self):
        ag = panel._pairwise_agreement([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        assert ag == pytest.approx(-1.0)


class FakePanelClient:
    """Scores answers by their position via a fixed mapping embedded as text."""
    def __init__(self, scores_by_marker):
        self.scores_by_marker = scores_by_marker

    async def generate(self, prompt, system_prompt=None, max_tokens=512,
                       temperature=1.0, **kwargs):
        # Determine n from the answers present
        import json
        # Build scores dict by checking which markers appear under which "Answer i"
        # Simpler: parse the answer order from the prompt.
        scores = {}
        for i in range(10):
            marker = f"### Answer {i}\n"
            if marker in prompt:
                # find the answer text after the marker
                seg = prompt.split(marker, 1)[1]
                text = seg.split("\n\n", 1)[0]
                for key, val in self.scores_by_marker.items():
                    if key in text:
                        scores[str(i)] = val
                        break
                else:
                    scores[str(i)] = 50
        return json.dumps({"scores": scores})


class TestRankAnswersHolistic:
    def test_aggregates_and_ranks(self):
        # GOOD should rank above HACK across a 2-judge panel
        clients = [
            FakePanelClient({"GOOD": 90, "HACK": 15}),
            FakePanelClient({"GOOD": 85, "HACK": 20}),
        ]
        sem = asyncio.Semaphore(4)
        result = asyncio.run(panel.rank_answers_holistic(
            clients, "question",
            ["this is a GOOD answer", "this is a HACK answer"],
            sem,
        ))
        assert result["n_judges_ok"] == 2
        # anchor_scores higher = better -> GOOD (index 0) should exceed HACK
        assert result["anchor_scores"][0] > result["anchor_scores"][1]
        # ranks: GOOD should be rank 0
        assert result["anchor_ranks"][0] < result["anchor_ranks"][1]
        assert result["inter_judge_agreement"] == pytest.approx(1.0)

    def test_all_judges_fail_graceful(self):
        class DeadClient:
            async def generate(self, *a, **k):
                raise RuntimeError("down")
        sem = asyncio.Semaphore(2)
        result = asyncio.run(panel.rank_answers_holistic(
            [DeadClient()], "q", ["a", "b"], sem, timeout=1.0,
        ))
        assert result["n_judges_ok"] == 0
        assert result["anchor_scores"] == [0.0, 0.0]
