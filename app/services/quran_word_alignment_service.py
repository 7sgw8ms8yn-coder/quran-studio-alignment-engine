from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from app.services.quran_candidate_matcher import (
    normalise_for_matching,
)
from app.services.speech_recognition_service import TranscriptWord


_ARABIC_LETTER_PATTERN = re.compile(r"[\u0621-\u064A]")


@dataclass(frozen=True, slots=True)
class TimedVerifiedWord:
    verified_index: int
    text: str
    start: float
    end: float
    confidence: float
    recognised_text: str


@dataclass(frozen=True, slots=True)
class QuranWordAlignmentResult:
    words: tuple[TimedVerifiedWord, ...]
    verified_word_count: int
    recognised_word_count: int
    matched_word_count: int
    recognised_coverage: float
    verified_coverage: float
    average_confidence: float


class QuranWordAlignmentService:
    def __init__(
        self,
        *,
        minimum_similarity: float = 0.58,
        skip_verified_penalty: float = 0.30,
        skip_recognised_penalty: float = 0.45,
    ) -> None:
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError(
                "minimum_similarity must be between 0 and 1."
            )

        self.minimum_similarity = minimum_similarity
        self.skip_verified_penalty = skip_verified_penalty
        self.skip_recognised_penalty = skip_recognised_penalty

    @staticmethod
    def extract_spoken_words(
        verified_text: str,
    ) -> tuple[str, ...]:
        return tuple(
            token
            for token in verified_text.split()
            if _ARABIC_LETTER_PATTERN.search(token)
        )

    @staticmethod
    def _similarity(
        recognised: str,
        verified: str,
    ) -> float:
        recognised_normalised = normalise_for_matching(
            recognised
        )
        verified_normalised = normalise_for_matching(
            verified
        )

        if not recognised_normalised or not verified_normalised:
            return 0.0

        if recognised_normalised == verified_normalised:
            return 1.0

        return SequenceMatcher(
            None,
            recognised_normalised,
            verified_normalised,
            autojunk=False,
        ).ratio()

    def align(
        self,
        recognised_words: Sequence[TranscriptWord],
        verified_text: str,
    ) -> QuranWordAlignmentResult:
        verified_words = self.extract_spoken_words(
            verified_text
        )
        recognised = tuple(
            word
            for word in recognised_words
            if word.text.strip()
        )

        recognised_count = len(recognised)
        verified_count = len(verified_words)

        if recognised_count == 0 or verified_count == 0:
            return QuranWordAlignmentResult(
                words=(),
                verified_word_count=verified_count,
                recognised_word_count=recognised_count,
                matched_word_count=0,
                recognised_coverage=0.0,
                verified_coverage=0.0,
                average_confidence=0.0,
            )

        # Dynamic programming with free leading/trailing verified words.
        # This allows alignment of a partial recitation inside a long ayah.
        scores = [
            [0.0] * (verified_count + 1)
            for _ in range(recognised_count + 1)
        ]
        actions = [
            [""] * (verified_count + 1)
            for _ in range(recognised_count + 1)
        ]

        for recognised_index in range(1, recognised_count + 1):
            scores[recognised_index][0] = (
                scores[recognised_index - 1][0]
                - self.skip_recognised_penalty
            )
            actions[recognised_index][0] = "skip_recognised"

        for verified_index in range(1, verified_count + 1):
            scores[0][verified_index] = 0.0
            actions[0][verified_index] = "skip_verified"

        for recognised_index in range(1, recognised_count + 1):
            recognised_word = recognised[recognised_index - 1]

            for verified_index in range(1, verified_count + 1):
                verified_word = verified_words[verified_index - 1]

                similarity = self._similarity(
                    recognised_word.text,
                    verified_word,
                )

                match_score = (
                    scores[recognised_index - 1][verified_index - 1]
                    + similarity
                )
                skip_recognised_score = (
                    scores[recognised_index - 1][verified_index]
                    - self.skip_recognised_penalty
                )
                skip_verified_score = (
                    scores[recognised_index][verified_index - 1]
                    - self.skip_verified_penalty
                )

                best_score = max(
                    match_score,
                    skip_recognised_score,
                    skip_verified_score,
                )

                scores[recognised_index][verified_index] = (
                    best_score
                )

                if best_score == match_score:
                    actions[recognised_index][verified_index] = (
                        "match"
                    )
                elif best_score == skip_recognised_score:
                    actions[recognised_index][verified_index] = (
                        "skip_recognised"
                    )
                else:
                    actions[recognised_index][verified_index] = (
                        "skip_verified"
                    )

        # Trailing verified words are free because the uploaded audio may
        # contain only the beginning or middle of an ayah.
        end_verified_index = max(
            range(verified_count + 1),
            key=lambda index: scores[recognised_count][index],
        )

        recognised_index = recognised_count
        verified_index = end_verified_index
        aligned_reversed: list[TimedVerifiedWord] = []

        while recognised_index > 0 and verified_index > 0:
            action = actions[recognised_index][verified_index]

            if action == "match":
                recognised_word = recognised[recognised_index - 1]
                verified_word = verified_words[verified_index - 1]

                similarity = self._similarity(
                    recognised_word.text,
                    verified_word,
                )

                if similarity >= self.minimum_similarity:
                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            similarity
                            * max(
                                0.0,
                                min(1.0, recognised_word.probability),
                            ),
                        ),
                    )

                    aligned_reversed.append(
                        TimedVerifiedWord(
                            verified_index=verified_index,
                            text=verified_word,
                            start=round(recognised_word.start, 3),
                            end=round(recognised_word.end, 3),
                            confidence=round(confidence, 6),
                            recognised_text=recognised_word.text,
                        )
                    )

                recognised_index -= 1
                verified_index -= 1
            elif action == "skip_recognised":
                recognised_index -= 1
            elif action == "skip_verified":
                verified_index -= 1
            else:
                break

        aligned_words = tuple(reversed(aligned_reversed))
        matched_count = len(aligned_words)

        average_confidence = (
            sum(word.confidence for word in aligned_words)
            / matched_count
            if matched_count
            else 0.0
        )

        return QuranWordAlignmentResult(
            words=aligned_words,
            verified_word_count=verified_count,
            recognised_word_count=recognised_count,
            matched_word_count=matched_count,
            recognised_coverage=round(
                matched_count / recognised_count,
                6,
            ),
            verified_coverage=round(
                matched_count / verified_count,
                6,
            ),
            average_confidence=round(
                average_confidence,
                6,
            ),
        )
