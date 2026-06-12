"""Tests for binary (HealthBench-protocol) rubric grading — CHG-021.

Covers:
    - parse_rubric_text: model-generated text -> structured items
    - aggregate_binary: HealthBench aggregation incl. negative points
    - parse_criteria_met: grader response parsing
    - Judge.evaluate_answers_binary: end-to-end with a fake client
"""

import asyncio
import json

import pytest

from grubrics_science.judge.binary import (
    aggregate_binary,
    build_grader_prompt,
    parse_criteria_met,
    parse_rubric_text,
)
from grubrics_science.judge.judge import Judge


# ---------------------------------------------------------------------------
# parse_rubric_text
# ---------------------------------------------------------------------------

class TestParseRubricText:
    def test_canonical_format(self):
        text = (
            "Points: 3, Item: Mentions glucose control and regular monitoring\n"
            "Points: 2, Item: Recommends diet and exercise modifications\n"
            "Points: -2, Item: Suggests stopping medication without consulting"
        )
        items = parse_rubric_text(text)
        assert len(items) == 3
        assert items[0] == {
            "points": 3.0,
            "criterion": "Mentions glucose control and regular monitoring",
        }
        assert items[2]["points"] == -2.0

    def test_bullets_and_numbering(self):
        text = (
            "- Points: 3, Item: First criterion\n"
            "* Points: 2, Item: Second criterion\n"
            "1. Points: 1, Item: Third criterion\n"
            "2) Points: 4, Item: Fourth criterion"
        )
        items = parse_rubric_text(text)
        assert [it["points"] for it in items] == [3.0, 2.0, 1.0, 4.0]

    def test_reversed_field_order(self):
        text = "Item: Checks vital signs first, Points: 5"
        items = parse_rubric_text(text)
        assert items == [{"points": 5.0, "criterion": "Checks vital signs first"}]

    def test_continuation_lines(self):
        text = (
            "Points: 3, Item: Recommends immediate ECG\n"
            "and cardiac enzyme testing to rule out\n"
            "myocardial infarction\n"
            "\n"
            "Points: 2, Item: Mentions aspirin"
        )
        items = parse_rubric_text(text)
        assert len(items) == 2
        assert "cardiac enzyme testing" in items[0]["criterion"]
        assert "myocardial infarction" in items[0]["criterion"]
        assert items[1]["criterion"] == "Mentions aspirin"

    def test_prose_preamble_ignored(self):
        text = (
            "Here is a rubric for evaluating the answer:\n"
            "\n"
            "Points: 3, Item: Correctly identifies the diagnosis"
        )
        items = parse_rubric_text(text)
        assert len(items) == 1
        assert items[0]["criterion"] == "Correctly identifies the diagnosis"

    def test_decimal_and_case_variants(self):
        text = "points: 2.5, item: Half-point criterion"
        items = parse_rubric_text(text)
        assert items == [{"points": 2.5, "criterion": "Half-point criterion"}]

    def test_garbage_returns_empty(self):
        assert parse_rubric_text("") == []
        assert parse_rubric_text("   \n\n  ") == []
        assert parse_rubric_text("This is just prose without any rubric items.") == []

    def test_item_without_text_dropped(self):
        text = "Points: 3, Item:\nPoints: 2, Item: Valid one"
        items = parse_rubric_text(text)
        # First item has no text on its line; the second line starts a new item,
        # so the empty first item is dropped.
        assert len(items) == 1
        assert items[0]["criterion"] == "Valid one"


# ---------------------------------------------------------------------------
# aggregate_binary
# ---------------------------------------------------------------------------

class TestAggregateBinary:
    def test_all_met(self):
        items = [
            {"points": 3, "criterion": "a"},
            {"points": 2, "criterion": "b"},
        ]
        result = aggregate_binary(items, [True, True])
        assert result["score"] == pytest.approx(1.0)
        assert result["achieved_points"] == 5
        assert result["total_possible"] == 5

    def test_partial(self):
        items = [
            {"points": 3, "criterion": "a"},
            {"points": 2, "criterion": "b"},
        ]
        result = aggregate_binary(items, [True, False])
        assert result["score"] == pytest.approx(3 / 5)

    def test_negative_criterion_met_subtracts(self):
        items = [
            {"points": 4, "criterion": "good thing"},
            {"points": -2, "criterion": "bad thing"},
        ]
        # Negative criterion met -> its points are added (i.e. subtract from score),
        # but total_possible counts only positive points (HealthBench formula).
        result = aggregate_binary(items, [True, True])
        assert result["total_possible"] == 4
        assert result["achieved_points"] == 2
        assert result["score"] == pytest.approx(0.5)

    def test_negative_only_can_go_below_zero(self):
        items = [
            {"points": 2, "criterion": "good"},
            {"points": -3, "criterion": "bad"},
        ]
        result = aggregate_binary(items, [False, True])
        assert result["score"] == pytest.approx(-1.5)

    def test_parse_failure_counted_not_met(self):
        items = [
            {"points": 3, "criterion": "a"},
            {"points": 2, "criterion": "b"},
        ]
        result = aggregate_binary(items, [True, None])
        assert result["score"] == pytest.approx(3 / 5)
        assert result["parse_failures"] == 1

    def test_no_positive_points(self):
        items = [{"points": -2, "criterion": "bad"}]
        result = aggregate_binary(items, [True])
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# parse_criteria_met
# ---------------------------------------------------------------------------

class TestParseCriteriaMet:
    def test_code_block(self):
        response = '```json\n{"explanation": "ok", "criteria_met": true}\n```'
        assert parse_criteria_met(response) is True

    def test_raw_json(self):
        response = '{"explanation": "nope", "criteria_met": false}'
        assert parse_criteria_met(response) is False

    def test_garbage_returns_none(self):
        assert parse_criteria_met("I think the answer is good.") is None
        assert parse_criteria_met("") is None
        assert parse_criteria_met('{"criteria_met": "yes"}') is None  # non-boolean


# ---------------------------------------------------------------------------
# Judge.evaluate_answers_binary end-to-end (fake client, no API)
# ---------------------------------------------------------------------------

class FakeBinaryClient:
    """Fake LLM client that grades by substring rules embedded in the prompt.

    Rules map (answer_marker, criterion_marker) -> criteria_met. The grader
    prompt contains both the conversation (with the answer) and the rubric
    item, so we match on substrings.
    """

    def __init__(self, rules, garbage_first_n=0):
        self.rules = rules
        self.calls = 0
        self._garbage_remaining = garbage_first_n

    async def generate(self, prompt, system_prompt=None, max_tokens=512,
                       temperature=1.0, **kwargs):
        self.calls += 1
        if self._garbage_remaining > 0:
            self._garbage_remaining -= 1
            return "Sorry, I cannot produce JSON right now."
        for (answer_marker, criterion_marker), met in self.rules.items():
            if answer_marker in prompt and criterion_marker in prompt:
                return json.dumps({"explanation": "test", "criteria_met": met})
        return json.dumps({"explanation": "default", "criteria_met": False})


def _make_judge(client):
    return Judge(client=client, max_concurrent=4, max_retries=2, timeout=30.0)


class TestEvaluateAnswersBinary:
    def test_scores_from_text_rubric(self):
        rubric_text = (
            "Points: 3, Item: Mentions ECG testing\n"
            "Points: 2, Item: Recommends aspirin"
        )
        rules = {
            ("ANSWER_GOOD", "ECG"): True,
            ("ANSWER_GOOD", "aspirin"): True,
            ("ANSWER_BAD", "ECG"): False,
            ("ANSWER_BAD", "aspirin"): True,
        }
        client = FakeBinaryClient(rules)
        judge = _make_judge(client)

        scores = asyncio.run(judge.evaluate_answers_binary(
            question="Chest pain, what do I do?",
            answers=["ANSWER_GOOD response", "ANSWER_BAD response"],
            rubric=rubric_text,
        ))

        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(2 / 5)
        # 2 answers x 2 criteria = 4 calls
        assert client.calls == 4

    def test_structured_rubric_input(self):
        items = [
            {"points": 4, "criterion": "Identifies the diagnosis"},
            {"points": -2, "criterion": "Recommends harmful action"},
        ]
        rules = {
            ("ANSWER_X", "diagnosis"): True,
            ("ANSWER_X", "harmful"): True,  # meets the negative criterion
        }
        client = FakeBinaryClient(rules)
        judge = _make_judge(client)

        scores, details = asyncio.run(judge.evaluate_answers_binary(
            question="q", answers=["ANSWER_X"], rubric=items,
            return_details=True,
        ))
        assert scores[0] == pytest.approx(0.5)  # (4 - 2) / 4
        assert details[0]["num_criteria"] == 2

    def test_unparseable_rubric_returns_zeros_without_calls(self):
        client = FakeBinaryClient({})
        judge = _make_judge(client)
        scores = asyncio.run(judge.evaluate_answers_binary(
            question="q", answers=["a1", "a2"], rubric="just prose, no items",
        ))
        assert scores == [0.0, 0.0]
        assert client.calls == 0

    def test_parse_retry_then_success(self):
        rubric_text = "Points: 1, Item: Mentions hydration"
        rules = {("ANSWER_H", "hydration"): True}
        client = FakeBinaryClient(rules, garbage_first_n=1)
        judge = _make_judge(client)

        scores = asyncio.run(judge.evaluate_answers_binary(
            question="q", answers=["ANSWER_H"], rubric=rubric_text,
        ))
        assert scores[0] == pytest.approx(1.0)
        assert client.calls == 2  # 1 garbage + 1 valid

    def test_prompt_messages_conversation(self):
        rubric_text = "Points: 1, Item: Mentions CONV_MARKER context"
        captured = {}

        class CapturingClient(FakeBinaryClient):
            async def generate(self, prompt, **kwargs):
                captured["prompt"] = prompt
                return await super().generate(prompt, **kwargs)

        client = CapturingClient({})
        judge = _make_judge(client)
        asyncio.run(judge.evaluate_answers_binary(
            question="ignored",
            answers=["the completion"],
            rubric=rubric_text,
            prompt_messages=[
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "second turn"},
                {"role": "user", "content": "third turn CONV_MARKER"},
            ],
        ))
        prompt = captured["prompt"]
        assert "user: first turn" in prompt
        assert "assistant: second turn" in prompt
        assert "CONV_MARKER" in prompt
        assert "assistant: the completion" in prompt


# ---------------------------------------------------------------------------
# build_grader_prompt
# ---------------------------------------------------------------------------

def test_build_grader_prompt_fills_template():
    prompt = build_grader_prompt("user: hi\n\nassistant: hello", "[3] greets the user")
    assert "user: hi" in prompt
    assert "[3] greets the user" in prompt
    assert "<<conversation>>" not in prompt
    assert "<<rubric_item>>" not in prompt
