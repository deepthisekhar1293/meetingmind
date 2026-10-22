"""Evaluate extraction quality against hand-labelled synthetic transcripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from extractor import DEFAULT_MODEL, MeetingNotes, extract_notes

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def contains(text: str, keyword: str) -> bool:
    """Check whether text contains a keyword without case-sensitive matching."""
    return keyword.casefold() in text.casefold()


def score_case(notes: MeetingNotes, case: dict[str, Any]) -> dict[str, int]:
    """Compare one extraction with its hand-labelled expected actions and tags."""
    expected = case["action_items"]
    found = owner_ok = 0
    matched_prediction_indexes: set[int] = set()

    for expected_item in expected:
        for index, prediction in enumerate(notes.action_items):
            if index in matched_prediction_indexes:
                continue
            if contains(prediction.task, expected_item["keyword"]):
                found += 1
                matched_prediction_indexes.add(index)
                if expected_item["owner"].casefold() in prediction.owner.casefold():
                    owner_ok += 1
                break

    decisions_found = sum(
        any(contains(decision, keyword) for decision in notes.decisions)
        for keyword in case["decision_keywords"]
    )
    risks_found = sum(
        any(contains(risk, keyword) for risk in notes.risks_or_blockers)
        for keyword in case["risk_keywords"]
    )
    return {
        "expected_actions": len(expected),
        "predicted_actions": len(notes.action_items),
        "found": found,
        "owner_ok": owner_ok,
        "decisions_expected": len(case["decision_keywords"]),
        "decisions_found": decisions_found,
        "risks_expected": len(case["risk_keywords"]),
        "risks_found": risks_found,
    }


def run(
    client: Any,
    cases: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> dict[str, int]:
    """Run Claude on each labelled transcript and print combined quality scores.

    This calls the paid API. The keyword-based scores are approximate and depend
    on the labels and predictions containing matching words.
    """
    totals = {
        "expected_actions": 0,
        "predicted_actions": 0,
        "found": 0,
        "owner_ok": 0,
        "decisions_expected": 0,
        "decisions_found": 0,
        "risks_expected": 0,
        "risks_found": 0,
    }
    for case in cases:
        transcript_path = (PROJECT_DIR / case["file"]).resolve()
        if not transcript_path.is_relative_to(PROJECT_DIR):
            raise ValueError(f"Test case path must stay within the project: {case['file']}")
        transcript = transcript_path.read_text(encoding="utf-8")
        notes = extract_notes(client, transcript, model=model)
        scores = score_case(notes, case)
        print(f"\n{case['file']}")
        print(
            f"  actions found {scores['found']}/{scores['expected_actions']} | "
            f"owners right {scores['owner_ok']}/{scores['found']} | "
            f"AI listed {scores['predicted_actions']} items | "
            f"decisions {scores['decisions_found']}/{scores['decisions_expected']}"
        )
        for key, value in scores.items():
            totals[key] += value

    def percentage(numerator: int, denominator: int) -> str:
        """Format a score as a percentage, or n/a when there is no denominator."""
        return f"{100 * numerator / denominator:.0f}%" if denominator else "n/a"

    print("\n=== OVERALL ===")
    print(f"Action item recall : {percentage(totals['found'], totals['expected_actions'])}")
    print(f"Owner accuracy     : {percentage(totals['owner_ok'], totals['found'])}")
    print(f"Precision          : {percentage(totals['found'], totals['predicted_actions'])}")
    print(
        f"Decision recall    : "
        f"{percentage(totals['decisions_found'], totals['decisions_expected'])}"
    )
    print(f"Risk recall        : {percentage(totals['risks_found'], totals['risks_expected'])}")
    return totals


def main() -> None:
    """Load local evaluation cases, create the Claude client, and run evaluation."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your .env file first.")
    cases_path = PROJECT_DIR / "sample_data" / "test_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    import anthropic

    run(anthropic.Anthropic(), cases)


if __name__ == "__main__":
    main()
