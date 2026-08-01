from pathlib import Path

import pytest

from src.ai.intent_profiles import INTENT_TARGETS, map_intent_to_profile
from src.recommender import load_songs

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "songs.csv"


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(CATALOG_PATH))


@pytest.mark.parametrize("intent", sorted(INTENT_TARGETS))
def test_every_accepted_intent_maps_to_a_valid_profile(catalog, intent):
    mapping = map_intent_to_profile(intent, catalog)
    assert mapping.intent == intent
    assert mapping.user_profile["favorite_genre"] == mapping.selected_genre
    assert mapping.user_profile["favorite_mood"] == mapping.selected_mood
    assert mapping.user_profile["preferred_energy"] == INTENT_TARGETS[intent]["energy"]
    assert mapping.user_profile["preferred_danceability"] == INTENT_TARGETS[intent]["danceability"]
    assert set(mapping.user_profile.keys()) == {
        "favorite_genre",
        "favorite_mood",
        "preferred_energy",
        "preferred_danceability",
    }


@pytest.mark.parametrize("intent", sorted(INTENT_TARGETS))
def test_selected_genre_and_mood_exist_in_catalog(catalog, intent):
    catalog_genres = {song["genre"].strip().lower() for song in catalog if song["genre"].strip()}
    catalog_moods = {song["mood"].strip().lower() for song in catalog if song["mood"].strip()}

    mapping = map_intent_to_profile(intent, catalog)

    assert mapping.selected_genre.lower() in catalog_genres
    assert mapping.selected_mood.lower() in catalog_moods


@pytest.mark.parametrize("intent", sorted(INTENT_TARGETS))
def test_numeric_targets_match_fixed_table(catalog, intent):
    mapping = map_intent_to_profile(intent, catalog)
    assert mapping.energy_target == INTENT_TARGETS[intent]["energy"]
    assert mapping.danceability_target == INTENT_TARGETS[intent]["danceability"]


def test_out_of_scope_cannot_be_mapped(catalog):
    with pytest.raises(ValueError):
        map_intent_to_profile("out_of_scope", catalog)


def test_none_intent_cannot_be_mapped(catalog):
    with pytest.raises(ValueError):
        map_intent_to_profile(None, catalog)


def test_deterministic_fallback_when_no_priority_value_is_present():
    songs = [
        {"genre": "zzz-unlisted-genre", "mood": "zzz-unlisted-mood"},
        {"genre": "zzz-unlisted-genre", "mood": "zzz-unlisted-mood"},
        {"genre": "aaa-other-genre", "mood": "aaa-other-mood"},
    ]

    mapping = map_intent_to_profile("workout", songs)

    assert mapping.genre_fallback_used is True
    assert mapping.mood_fallback_used is True
    assert mapping.selected_genre == "zzz-unlisted-genre"
    assert mapping.selected_mood == "zzz-unlisted-mood"


def test_fallback_breaks_frequency_ties_alphabetically_case_insensitive():
    songs = [
        {"genre": "Zeta-genre", "mood": "Zeta-mood"},
        {"genre": "alpha-genre", "mood": "alpha-mood"},
    ]

    mapping = map_intent_to_profile("workout", songs)

    assert mapping.genre_fallback_used is True
    assert mapping.selected_genre == "alpha-genre"
    assert mapping.selected_mood == "alpha-mood"


def test_repeated_calls_produce_the_same_fallback_result():
    songs = [
        {"genre": "custom-genre-a", "mood": "custom-mood-a"},
        {"genre": "custom-genre-b", "mood": "custom-mood-b"},
        {"genre": "custom-genre-b", "mood": "custom-mood-b"},
    ]
    first = map_intent_to_profile("sleep", songs)
    second = map_intent_to_profile("sleep", songs)
    assert first.selected_genre == second.selected_genre
    assert first.selected_mood == second.selected_mood


def test_no_invented_catalog_value(catalog):
    catalog_genres = {song["genre"].strip().lower() for song in catalog if song["genre"].strip()}
    catalog_moods = {song["mood"].strip().lower() for song in catalog if song["mood"].strip()}
    for intent in INTENT_TARGETS:
        mapping = map_intent_to_profile(intent, catalog)
        assert mapping.selected_genre.lower() in catalog_genres
        assert mapping.selected_mood.lower() in catalog_moods


def test_empty_catalog_raises_value_error():
    with pytest.raises(ValueError):
        map_intent_to_profile("workout", [])


def test_catalog_with_only_blank_genre_raises_value_error():
    songs = [{"genre": "   ", "mood": "happy"}]
    with pytest.raises(ValueError):
        map_intent_to_profile("workout", songs)


def test_catalog_with_only_blank_mood_raises_value_error():
    songs = [{"genre": "pop", "mood": "   "}]
    with pytest.raises(ValueError):
        map_intent_to_profile("workout", songs)
