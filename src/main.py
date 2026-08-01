"""Command line runner for the Music Recommender Simulation."""

import argparse
import logging
from typing import List, Optional

from src.ai.intent_classifier import IntentClassifier
from src.ai.intent_profiles import map_intent_to_profile
from src.recommender import load_songs, recommend_songs

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CATALOG_PATH = "data/songs.csv"


def main() -> None:
    songs = load_songs("data/songs.csv")

    default_user_profile = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "preferred_energy": 0.80,
        "preferred_danceability": 0.80,
    }

    recommendations = recommend_songs(default_user_profile, songs, k=5)

    print("\nTop recommendations:\n")
    for rank, recommendation in enumerate(recommendations, start=1):
        song = recommendation["song"]
        score = recommendation["score"]
        reasons = recommendation["reasons"]
        print(f"{rank}. {song['title']} — {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"   Score: {score:.2f}/10")
        print("   Reasons:")
        for reason in reasons:
            print(f"   - {reason}")
        print()


def _top_k_type(raw_value: str) -> int:
    """argparse type hook enforcing an integer top-k between 1 and 10."""
    try:
        value = int(raw_value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--top-k must be an integer, got '{raw_value}'")
    if value < 1 or value > 10:
        raise argparse.ArgumentTypeError(f"--top-k must be between 1 and 10, got {value}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the AI-assisted recommendation workflow."""
    parser = argparse.ArgumentParser(
        description="Intentune: AI-Assisted Music Recommendation System"
    )
    parser.add_argument(
        "--request",
        type=str,
        default=None,
        help="Natural-language music request, e.g. 'I need energetic music for a workout'",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=_top_k_type,
        default=DEFAULT_TOP_K,
        help="Number of recommendations to return (integer, 1-10, default 5)",
    )
    return parser


def run(argv: Optional[List[str]] = None) -> int:
    """Run the natural-language intent -> structured profile -> recommendation workflow.

    Returns a process exit status: 0 on a successfully served request, 1 when
    the request is safely rejected by a guardrail or a handled application
    error occurs. Never lets an expected condition raise an unhandled
    traceback out of this function.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    request_text = args.request
    if request_text is None:
        request_text = input("Enter a natural-language music request: ")

    print("Intentune: AI-Assisted Music Recommendation System")
    print()
    print("Request:")
    print(request_text)
    print()

    classifier = IntentClassifier().train()
    prediction = classifier.predict(request_text)

    logger.info(
        "request length=%d accepted=%s intent=%s confidence=%.3f",
        len(request_text),
        prediction.accepted,
        prediction.intent,
        prediction.confidence,
    )

    if not prediction.accepted:
        print("AI interpretation:")
        print("Request rejected.")
        print(prediction.reason)
        return 1

    try:
        songs = load_songs(DEFAULT_CATALOG_PATH)
        mapping = map_intent_to_profile(prediction.intent, songs)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("handled application error: %s", exc)
        print(f"Could not generate recommendations: {exc}")
        return 1

    print("AI interpretation:")
    print(f"Intent: {mapping.intent}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Genre target: {mapping.selected_genre}")
    print(f"Mood target: {mapping.selected_mood}")
    print(f"Energy target: {mapping.energy_target:.2f}")
    print(f"Danceability target: {mapping.danceability_target:.2f}")
    print()

    recommendations = recommend_songs(mapping.user_profile, songs, k=args.top_k)
    logger.info("recommendation count=%d", len(recommendations))

    print("Recommendations:")
    for rank, recommendation in enumerate(recommendations, start=1):
        song = recommendation["song"]
        score = recommendation["score"]
        reasons = recommendation["reasons"]
        print(f"{rank}. {song['title']} — {song['artist']}")
        print(f"   Score: {score:.2f}/10")
        print("   Reasons:")
        for reason in reasons:
            print(f"   - {reason}")
        print()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(run())
