"""Unified training entry point for digit and character models."""

from __future__ import annotations

import argparse

from train_character import train_character_model
from train_digit import train_digit_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train handwritten character recognition models."
    )
    parser.add_argument(
        "--model",
        choices=["digit", "character", "both"],
        default="both",
        help="Which model to train (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    """Train the selected model(s)."""
    args = parse_args()

    if args.model in ("digit", "both"):
        train_digit_model()

    if args.model in ("character", "both"):
        train_character_model()


if __name__ == "__main__":
    main()
