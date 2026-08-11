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
    surah_number: int | None = None
    ayah_number: int | None = None
    word_index_in_ayah: int | None = None


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
        minimum_confidence: float = 0.60,
        minimum_duration: float = 0.08,
        timestamp_tolerance: float = 0.03,
        skip_verified_penalty: float = 0.30,
        skip_recognised_penalty: float = 0.45,
    ) -> None:
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError(
                "minimum_similarity must be between 0 and 1."
            )

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if minimum_duration < 0.0:
            raise ValueError(
                "minimum_duration cannot be negative."
            )

        if timestamp_tolerance < 0.0:
            raise ValueError(
                "timestamp_tolerance cannot be negative."
            )

        self.minimum_similarity = minimum_similarity
        self.minimum_confidence = minimum_confidence
        self.minimum_duration = minimum_duration
        self.timestamp_tolerance = timestamp_tolerance
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

                    start = float(recognised_word.start)
                    end = float(recognised_word.end)
                    duration = end - start

                    timestamp_is_valid = (
                        start >= 0.0
                        and end > start
                        and duration >= self.minimum_duration
                    )

                    quality_is_valid = (
                        confidence >= self.minimum_confidence
                    )

                    if timestamp_is_valid and quality_is_valid:
                        aligned_reversed.append(
                            TimedVerifiedWord(
                                verified_index=verified_index,
                                text=verified_word,
                                start=round(start, 3),
                                end=round(end, 3),
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

        ordered_words = tuple(reversed(aligned_reversed))

        # Reject timestamps that move backwards or substantially overlap.
        # Small model rounding differences are tolerated.
        validated_words: list[TimedVerifiedWord] = []
        previous_end = 0.0

        for word in ordered_words:
            if (
                validated_words
                and word.start
                < previous_end - self.timestamp_tolerance
            ):
                continue

            validated_words.append(word)
            previous_end = max(previous_end, word.end)

        # Narrow recovery for one missing FINAL Quran word.
        #
        # Whisper can occasionally fragment or slightly misrecognise the
        # final recited word. We must NOT lower the global alignment
        # thresholds because that would weaken Quran matching everywhere.
        #
        # Recovery is allowed only when:
        #   1. We already have a valid aligned sequence.
        #   2. Exactly ONE verified Quran word remains.
        #   3. Whisper produced immediate trailing speech.
        #   4. That speech has moderate textual evidence for the final word.
        #
        # The recovered word keeps a LOW confidence value so downstream
        # consumers can still distinguish recovery from a normal match.

        if validated_words:
            last_aligned = validated_words[-1]

            remaining_verified_words = (
                verified_count - last_aligned.verified_index
            )

            if remaining_verified_words == 1:
                final_verified_word = verified_words[-1]

                trailing_recognised = tuple(
                    word
                    for word in recognised
                    if (
                        word.text.strip()
                        and float(word.end) > last_aligned.end
                        and float(word.start)
                        >= (
                            last_aligned.end
                            - self.timestamp_tolerance
                        )
                        and float(word.start)
                        <= last_aligned.end + 3.0
                        and (
                            float(word.end)
                            - float(word.start)
                        )
                        >= self.minimum_duration
                    )
                )

                if trailing_recognised:
                    trailing_evidence = tuple(
                        (
                            word,
                            self._similarity(
                                word.text,
                                final_verified_word,
                            ),
                        )
                        for word in trailing_recognised
                    )

                    best_similarity = max(
                        similarity
                        for _, similarity in trailing_evidence
                    )

                    best_combined_confidence = max(
                        similarity
                        * max(
                            0.0,
                            min(
                                1.0,
                                float(word.probability),
                            ),
                        )
                        for word, similarity in trailing_evidence
                    )

                    first_tail_start = min(
                        float(word.start)
                        for word in trailing_recognised
                    )

                    tail_is_immediate = (
                        first_tail_start
                        <= last_aligned.end + 0.75
                    )

                    if (
                        tail_is_immediate
                        and best_similarity >= 0.45
                        and best_combined_confidence >= 0.25
                    ):
                        recovery_start = max(
                            last_aligned.end,
                            first_tail_start,
                        )

                        recovery_end = max(
                            float(word.end)
                            for word in trailing_recognised
                        )

                        validated_words.append(
                            TimedVerifiedWord(
                                verified_index=verified_count,
                                text=final_verified_word,
                                start=round(
                                    recovery_start,
                                    3,
                                ),
                                end=round(
                                    recovery_end,
                                    3,
                                ),
                                confidence=round(
                                    best_combined_confidence,
                                    6,
                                ),
                                recognised_text=" ".join(
                                    word.text.strip()
                                    for word
                                    in trailing_recognised
                                ),
                            )
                        )

        aligned_words = tuple(validated_words)
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
