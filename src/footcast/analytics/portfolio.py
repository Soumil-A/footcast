"""Audited portfolio-facing evidence from the sealed 2024-25 model test."""

from __future__ import annotations

from typing import Any

TEST_SEASON = "2024-25"
TEST_MATCHES = 380

# Values are copied from reports/final_test_results.json. Keeping this compact
# presentation contract in code makes the deployed API image self-contained;
# tests guard the figures against accidental drift from the tracked report.
MODEL_BENCHMARKS: tuple[dict[str, Any], ...] = (
    {
        "model": "Majority class",
        "accuracy": 0.40789473684210525,
        "macro_f1": 0.19314641744548286,
        "log_loss": 1.0810879896180008,
    },
    {
        "model": "Elo (deployed)",
        "accuracy": 0.5263157894736842,
        "macro_f1": 0.39236626689510756,
        "log_loss": 0.9934579055185351,
    },
    {
        "model": "Logistic regression",
        "accuracy": 0.5289473684210526,
        "macro_f1": 0.3979068511974872,
        "log_loss": 1.0063091820486896,
    },
    {
        "model": "Frozen Random Forest",
        "accuracy": 0.5131578947368421,
        "macro_f1": 0.38314176245210724,
        "log_loss": 1.0049793479333773,
    },
)

DEPLOYED_ELO_RECALL = {
    "home_win": 0.8451612903225807,
    "draw": 0.0,
    "away_win": 0.5227272727272727,
}

DEPLOYED_ELO_CONFUSION_MATRIX = (
    (131, 0, 24),
    (63, 0, 30),
    (63, 0, 69),
)


def final_test_evidence() -> dict[str, Any]:
    """Return a copy-safe summary of the untouched final-test evidence."""
    return {
        "test_season": TEST_SEASON,
        "test_matches": TEST_MATCHES,
        "benchmarks": [dict(item) for item in MODEL_BENCHMARKS],
        "deployed_elo_recall": dict(DEPLOYED_ELO_RECALL),
        "deployed_elo_confusion_matrix": [
            list(row) for row in DEPLOYED_ELO_CONFUSION_MATRIX
        ],
        "class_order": ["home_win", "draw", "away_win"],
        "selection_note": (
            "The simpler Elo reference was deployed because the frozen Random "
            "Forest did not show a stable test advantage."
        ),
    }
