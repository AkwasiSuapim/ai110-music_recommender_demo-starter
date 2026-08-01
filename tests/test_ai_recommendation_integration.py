from pathlib import Path
from unittest.mock import patch

import pytest

from src import main as main_module
from src.ai.intent_profiles import map_intent_to_profile
from src.recommender import load_songs, recommend_songs

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"


def test_accepted_request_reaches_the_recommender(capsys):
    exit_code = main_module.run(["--request", "I need energetic music for a workout."])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "AI interpretation:" in output
    assert "Recommendations:" in output
    assert "Score:" in output


def test_rejected_request_does_not_call_recommendation_scoring(capsys):
    with patch("src.main.recommend_songs") as mock_recommend_songs:
        exit_code = main_module.run(["--request", "What is the weather tomorrow?"])

    assert exit_code == 1
    mock_recommend_songs.assert_not_called()
    output = capsys.readouterr().out
    assert "Recommendations:" not in output
    assert "Request rejected." in output


def test_top_k_is_honored(capsys):
    main_module.run(["--request", "I need energetic music for a workout.", "--top-k", "2"])
    output = capsys.readouterr().out
    assert output.count("Score:") == 2


def test_default_top_k_is_five(capsys):
    main_module.run(["--request", "I need energetic music for a workout."])
    output = capsys.readouterr().out
    assert output.count("Score:") == 5


def test_top_k_below_one_is_rejected():
    with pytest.raises(SystemExit) as exc_info:
        main_module.run(["--request", "test request text", "--top-k", "0"])
    assert exc_info.value.code != 0


def test_top_k_above_ten_is_rejected():
    with pytest.raises(SystemExit) as exc_info:
        main_module.run(["--request", "test request text", "--top-k", "11"])
    assert exc_info.value.code != 0


def test_top_k_non_integer_is_rejected():
    with pytest.raises(SystemExit) as exc_info:
        main_module.run(["--request", "test request text", "--top-k", "not-a-number"])
    assert exc_info.value.code != 0


def test_recommendation_result_includes_scores_and_explanations(capsys):
    main_module.run(["--request", "I need energetic music for a workout."])
    output = capsys.readouterr().out
    assert "Score:" in output
    assert "Reasons:" in output


def test_two_intents_produce_meaningfully_different_structured_profiles():
    songs = load_songs(str(CATALOG_PATH))
    workout_mapping = map_intent_to_profile("workout", songs)
    sleep_mapping = map_intent_to_profile("sleep", songs)

    assert workout_mapping.user_profile != sleep_mapping.user_profile
    assert workout_mapping.energy_target != sleep_mapping.energy_target
    assert workout_mapping.danceability_target != sleep_mapping.danceability_target


def test_two_intents_produce_different_recommendation_behavior():
    songs = load_songs(str(CATALOG_PATH))
    workout_mapping = map_intent_to_profile("workout", songs)
    sleep_mapping = map_intent_to_profile("sleep", songs)

    workout_top = recommend_songs(workout_mapping.user_profile, songs, k=1)[0]["song"]["title"]
    sleep_top = recommend_songs(sleep_mapping.user_profile, songs, k=1)[0]["song"]["title"]

    assert workout_top != sleep_top


def test_empty_request_is_handled_without_traceback(capsys):
    exit_code = main_module.run(["--request", ""])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Request rejected." in output
