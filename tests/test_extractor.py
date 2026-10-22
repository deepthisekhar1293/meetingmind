"""Unit tests for MeetingMind that do not make paid Claude API calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from evaluate import score_case
from extractor import (
    MAX_TRANSCRIPT_CHARS,
    MeetingNotes,
    extract_json,
    extract_notes,
    to_csv,
    to_jira_csv,
    to_markdown,
)

GOOD = {
    "summary": "Team agreed to launch on Nov 18.",
    "decisions": ["Launch on November 18", "Drop gift cards"],
    "action_items": [
        {
            "task": "Finish refund flow",
            "owner": "Rahul",
            "deadline": "Wednesday",
            "priority": "high",
        },
        {
            "task": "Write API docs",
            "owner": "Rahul",
            "deadline": "Not specified",
            "priority": "weird",
        },
    ],
    "risks_or_blockers": ["Only one tester"],
    "open_questions": ["Is legal review needed?"],
}
TRANSCRIPT = "Priya: Let's launch on Nov 18. Rahul: I will finish the refund flow by Wednesday."


class FakeClient:
    """Stand-in Claude client that returns prepared replies without API calls."""

    def __init__(self, replies: list[str]) -> None:
        """Store replies and capture requests so tests can inspect API behavior."""
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        """Mimic the Claude SDK create method with the next prepared reply."""
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
        )


def test_extract_json_plain_fenced_and_chatty() -> None:
    """Confirm JSON parsing supports the response formats the model may return."""
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": 1} Hope that helps') == {"a": 1}


@pytest.mark.parametrize("reply", ["no json here", "[]", "```json\nnot json\n```"])
def test_extract_json_rejects_invalid_replies(reply: str) -> None:
    """Confirm malformed or non-object model replies are rejected."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract_json(reply)


def test_extract_notes_happy_path_and_priority_normalised() -> None:
    """Check valid notes are returned and priorities use standard labels."""
    client = FakeClient([json.dumps(GOOD)])
    notes = extract_notes(client, TRANSCRIPT)

    assert isinstance(notes, MeetingNotes)
    assert notes.action_items[0].priority == "High"
    assert notes.action_items[1].priority == "Medium"
    assert len(client.calls) == 1
    assert client.calls[0]["model"]


def test_retry_repairs_invalid_output() -> None:
    """Check the extractor asks Claude for a corrected reply after bad JSON."""
    client = FakeClient(["this is not json", json.dumps(GOOD)])
    notes = extract_notes(client, TRANSCRIPT)

    assert notes.summary.startswith("Team agreed")
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "user"


def test_raises_after_retry_limit() -> None:
    """Check extraction reports an error if Claude remains invalid after retry."""
    client = FakeClient(["garbage"])
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        extract_notes(client, TRANSCRIPT)
    assert len(client.calls) == 2


def test_rejects_short_and_oversized_transcripts_without_api_call() -> None:
    """Check invalid transcript lengths are rejected before contacting Claude."""
    client = FakeClient([json.dumps(GOOD)])
    with pytest.raises(ValueError, match="too short"):
        extract_notes(client, "hi")
    with pytest.raises(ValueError, match="too long"):
        extract_notes(client, "x" * (MAX_TRANSCRIPT_CHARS + 1))
    assert client.calls == []


def test_optional_fields_default_correctly() -> None:
    """Check omitted owners, deadlines, and lists get safe default values."""
    minimal = {"summary": "A brief summary.", "action_items": [{"task": "Do thing"}]}
    notes = extract_notes(FakeClient([json.dumps(minimal)]), TRANSCRIPT)

    assert notes.action_items[0].owner == "Unassigned"
    assert notes.action_items[0].deadline == "Not specified"
    assert notes.decisions == []


def test_exports_include_expected_columns_and_content() -> None:
    """Check Markdown, regular CSV, and Jira CSV contain their expected fields."""
    notes = extract_notes(FakeClient([json.dumps(GOOD)]), TRANSCRIPT)

    assert "## Action Items" in to_markdown(notes)
    assert "Finish refund flow" in to_markdown(notes)
    assert to_csv(notes).splitlines()[0] == "Task,Owner,Deadline,Priority"
    assert to_jira_csv(notes).splitlines()[0] == (
        "Summary,Assignee,Priority,Description,Issue Type"
    )
    assert "Rahul" in to_jira_csv(notes)


def test_exports_handle_empty_action_items_and_markdown_table_delimiters() -> None:
    """Check empty exports work and Markdown table separators are escaped."""
    notes = MeetingNotes(
        summary="Summary",
        action_items=[
            {"task": "Review A | B", "owner": "Unassigned", "deadline": "Friday"}
        ],
    )

    assert "Review A \\| B" in to_markdown(notes)
    assert pd.read_csv(pd.io.common.StringIO(to_csv(MeetingNotes(summary="Summary")))).empty
    jira_frame = pd.read_csv(pd.io.common.StringIO(to_jira_csv(notes)))
    assert jira_frame.loc[0, "Assignee"] != jira_frame.loc[0, "Assignee"]


def test_rejects_negative_retry_count_without_api_call() -> None:
    """Check invalid retry settings fail before calling the Claude client."""
    client = FakeClient([json.dumps(GOOD)])
    with pytest.raises(ValueError, match="non-negative"):
        extract_notes(client, TRANSCRIPT, retries=-1)
    assert client.calls == []


def test_score_case_matches_each_prediction_once() -> None:
    """Check evaluation does not count one predicted action more than once."""
    notes = MeetingNotes(
        summary="Standup",
        decisions=["All API endpoints need tests"],
        action_items=[
            {"task": "Escalate staging crash", "owner": "Neha"},
            {"task": "Escalate staging issue again", "owner": "Sam"},
        ],
        risks_or_blockers=["Staging server keeps crashing"],
    )
    result = score_case(
        notes,
        {
            "action_items": [
                {"keyword": "staging", "owner": "Neha"},
                {"keyword": "staging", "owner": "Neha"},
            ],
            "decision_keywords": ["tests"],
            "risk_keywords": ["staging"],
        },
    )

    assert result["found"] == 2
    assert result["owner_ok"] == 1
    assert result["decisions_found"] == 1
    assert result["risks_found"] == 1
