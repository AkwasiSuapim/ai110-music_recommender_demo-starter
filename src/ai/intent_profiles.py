"""Maps an accepted music intent to a structured recommender user profile.

The intent classifier only decides *what the listener wants*; this module
turns that decision into the exact profile shape the deterministic
recommender (src/recommender.py) already expects, using catalog-aware
genre and mood selection so nothing is invented that the catalog does not
actually contain.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

INTENT_TARGETS: Dict[str, Dict[str, float]] = {
    "workout": {"energy": 0.90, "danceability": 0.85},
    "study": {"energy": 0.30, "danceability": 0.20},
    "relax": {"energy": 0.25, "danceability": 0.25},
    "party": {"energy": 0.90, "danceability": 0.95},
    "sleep": {"energy": 0.10, "danceability": 0.05},
    "mood_boost": {"energy": 0.75, "danceability": 0.70},
}

GENRE_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "workout": ("edm", "pop", "hip-hop", "afrobeats", "dance"),
    "study": ("classical", "lofi", "lo-fi", "acoustic", "instrumental", "jazz"),
    "relax": ("acoustic", "r&b", "classical", "jazz", "pop"),
    "party": ("edm", "dance", "pop", "afrobeats", "hip-hop"),
    "sleep": ("ambient", "classical", "acoustic", "instrumental", "jazz"),
    "mood_boost": ("pop", "afrobeats", "dance", "hip-hop", "r&b"),
}

MOOD_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "workout": ("energetic", "happy", "uplifting", "excited"),
    "study": ("calm", "focused", "relaxed", "peaceful"),
    "relax": ("relaxed", "calm", "peaceful", "chill"),
    "party": ("energetic", "happy", "excited", "uplifting"),
    "sleep": ("calm", "peaceful", "relaxed", "dreamy"),
    "mood_boost": ("happy", "uplifting", "energetic", "hopeful"),
}


@dataclass
class IntentProfileMapping:
    """Typed result of mapping one accepted intent onto the song catalog."""

    intent: str
    user_profile: Dict[str, object]
    selected_genre: str
    selected_mood: str
    energy_target: float
    danceability_target: float
    genre_fallback_used: bool
    mood_fallback_used: bool


def _nonblank_values(songs: List[Dict], field: str) -> List[str]:
    return [str(song[field]).strip() for song in songs if str(song.get(field, "")).strip()]


def _first_seen_casing(values: List[str]) -> Dict[str, str]:
    casing_map: Dict[str, str] = {}
    for value in values:
        key = value.lower()
        if key not in casing_map:
            casing_map[key] = value
    return casing_map


def _most_frequent_value(values: List[str]) -> str:
    lowered = [value.lower() for value in values]
    counts = Counter(lowered)
    casing_map = _first_seen_casing(values)
    max_count = max(counts.values())
    tied_keys = sorted(key for key, count in counts.items() if count == max_count)
    return casing_map[tied_keys[0]]


def _select_catalog_value(values: List[str], priority_list: Tuple[str, ...]) -> Tuple[str, bool]:
    """Pick the first priority-list value present in the catalog, else the most frequent one."""
    casing_map = _first_seen_casing(values)
    for candidate in priority_list:
        key = candidate.lower()
        if key in casing_map:
            return casing_map[key], False
    return _most_frequent_value(values), True


def map_intent_to_profile(intent: str, songs: List[Dict]) -> IntentProfileMapping:
    """Convert an accepted music intent into the recommender's exact profile shape.

    Raises ValueError if the intent is not one of the six mappable music
    intents (out_of_scope and rejected predictions must never reach this
    function), or if the catalog has no usable genre or mood values.
    """
    if intent not in INTENT_TARGETS:
        raise ValueError(
            f"Cannot map intent '{intent}' to a profile; expected one of "
            f"{sorted(INTENT_TARGETS)}"
        )

    genres = _nonblank_values(songs, "genre")
    moods = _nonblank_values(songs, "mood")

    if not genres:
        raise ValueError("Catalog contains no usable (nonblank) genre values")
    if not moods:
        raise ValueError("Catalog contains no usable (nonblank) mood values")

    selected_genre, genre_fallback_used = _select_catalog_value(genres, GENRE_PRIORITY[intent])
    selected_mood, mood_fallback_used = _select_catalog_value(moods, MOOD_PRIORITY[intent])

    targets = INTENT_TARGETS[intent]
    user_profile: Dict[str, object] = {
        "favorite_genre": selected_genre,
        "favorite_mood": selected_mood,
        "preferred_energy": targets["energy"],
        "preferred_danceability": targets["danceability"],
    }

    return IntentProfileMapping(
        intent=intent,
        user_profile=user_profile,
        selected_genre=selected_genre,
        selected_mood=selected_mood,
        energy_target=targets["energy"],
        danceability_target=targets["danceability"],
        genre_fallback_used=genre_fallback_used,
        mood_fallback_used=mood_fallback_used,
    )
