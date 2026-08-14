from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

from app.services.quran_word_alignment_service import (
    QuranWordAlignmentResult,
    QuranWordAlignmentService,
    TimedVerifiedWord,
)
from app.services.speech_recognition_service import (
    TranscriptWord,
    TranscriptionResult,
)


class RegionRecognizerProtocol(Protocol):
    def transcribe_region(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> TranscriptionResult:
        ...


@dataclass(frozen=True, slots=True)
class RepetitionRecoveryResult:
    recognised_words: tuple[TranscriptWord, ...]
    word_alignment: QuranWordAlignmentResult | None
    attempted_regions: int
    accepted_regions: int


class QuranRepetitionRecoveryService:
    """
    Guarded recovery for repeated Quran phrases that Whisper may collapse
    into one abnormally long word timestamp.

    Recovery is deliberately conservative:

    1. Only suspicious long recognised words are investigated.
    2. Only high-confidence micro-window words are considered.
    3. Recovered words may never extend beyond the collapsed region.
    4. The standard Quran word aligner performs final verification.
    5. Existing aligned words outside the repaired region must survive.
    6. A recovery is accepted only when Quran matching improves.
    """

    def __init__(
        self,
        *,
        recognizer: RegionRecognizerProtocol,
        word_aligner: QuranWordAlignmentService,
        suspicious_duration: float = 3.0,
        recovery_start_offset: float = 0.10,
        recovery_end_padding: float = 0.70,
        minimum_recovery_probability: float = 0.90,
        minimum_match_gain: int = 1,
        maximum_coverage_drop: float = 0.02,
        maximum_confidence_drop: float = 0.03,
        preserve_time_tolerance: float = 0.25,
        tiny_fragment_duration: float = 0.20,
        tiny_fragment_max_probability: float = 0.95,
    ) -> None:
        self.recognizer = recognizer
        self.word_aligner = word_aligner

        self.suspicious_duration = suspicious_duration
        self.recovery_start_offset = recovery_start_offset
        self.recovery_end_padding = recovery_end_padding
        self.minimum_recovery_probability = (
            minimum_recovery_probability
        )
        self.minimum_match_gain = minimum_match_gain
        self.maximum_coverage_drop = maximum_coverage_drop
        self.maximum_confidence_drop = maximum_confidence_drop
        self.preserve_time_tolerance = preserve_time_tolerance
        self.tiny_fragment_duration = tiny_fragment_duration
        self.tiny_fragment_max_probability = (
            tiny_fragment_max_probability
        )

    @staticmethod
    def _duration(word: TranscriptWord) -> float:
        return max(
            0.0,
            float(word.end) - float(word.start),
        )

    @staticmethod
    def _flatten_words(
        transcription: TranscriptionResult,
    ) -> tuple[TranscriptWord, ...]:
        return tuple(
            word
            for segment in transcription.segments
            for word in segment.words
        )

    def _is_protected_baseline_word(
        self,
        word: TimedVerifiedWord,
        collapsed: TranscriptWord,
    ) -> bool:
        """
        Words outside the suspicious collapsed interval are protected.

        A successful recovery must not make these words disappear.
        """
        return (
            float(word.end)
            <= float(collapsed.start)
            + self.preserve_time_tolerance
            or
            float(word.start)
            >= float(collapsed.end)
            - self.preserve_time_tolerance
        )

    def _baseline_word_survives(
        self,
        baseline_word: TimedVerifiedWord,
        candidate: QuranWordAlignmentResult,
    ) -> bool:
        """
        Repeated Quran indices may legitimately occur more than once,
        therefore preservation is checked using both verified index and
        timestamp proximity.
        """
        for candidate_word in candidate.words:
            if (
                candidate_word.verified_index
                != baseline_word.verified_index
            ):
                continue

            if (
                abs(
                    float(candidate_word.start)
                    - float(baseline_word.start)
                )
                <= self.preserve_time_tolerance
                and
                abs(
                    float(candidate_word.end)
                    - float(baseline_word.end)
                )
                <= self.preserve_time_tolerance
            ):
                return True

        return False

    def _preserves_existing_alignment(
        self,
        baseline: QuranWordAlignmentResult,
        candidate: QuranWordAlignmentResult,
        collapsed: TranscriptWord,
    ) -> bool:
        protected = tuple(
            word
            for word in baseline.words
            if self._is_protected_baseline_word(
                word,
                collapsed,
            )
        )

        return all(
            self._baseline_word_survives(
                word,
                candidate,
            )
            for word in protected
        )

    @staticmethod
    def _overlap_seconds(
        word: TimedVerifiedWord,
        collapsed: TranscriptWord,
    ) -> float:
        return max(
            0.0,
            min(
                float(word.end),
                float(collapsed.end),
            )
            - max(
                float(word.start),
                float(collapsed.start),
            ),
        )

    def _find_collapsed_anchor(
        self,
        baseline: QuranWordAlignmentResult,
        collapsed: TranscriptWord,
    ) -> TimedVerifiedWord | None:
        overlapping = [
            word
            for word in baseline.words
            if self._overlap_seconds(
                word,
                collapsed,
            ) > 0.0
        ]

        if not overlapping:
            return None

        return max(
            overlapping,
            key=lambda word: self._overlap_seconds(
                word,
                collapsed,
            ),
        )

    def _adds_anchored_repeat(
        self,
        baseline: QuranWordAlignmentResult,
        candidate: QuranWordAlignmentResult,
        collapsed: TranscriptWord,
    ) -> bool:
        """
        Recovery must add another occurrence of the Quran word
        that anchors the suspicious collapsed interval.

        This prevents unrelated micro-window speech from being
        accepted merely because it increases the global match score.
        """
        anchor = self._find_collapsed_anchor(
            baseline,
            collapsed,
        )

        if anchor is None:
            return False

        baseline_anchor_count = sum(
            1
            for word in baseline.words
            if (
                word.verified_index
                == anchor.verified_index
                and self._overlap_seconds(
                    word,
                    collapsed,
                ) > 0.0
            )
        )

        candidate_anchor_count = sum(
            1
            for word in candidate.words
            if (
                word.verified_index
                == anchor.verified_index
                and self._overlap_seconds(
                    word,
                    collapsed,
                ) > 0.0
            )
        )

        return (
            candidate_anchor_count
            > baseline_anchor_count
        )

    def _accept_candidate(
        self,
        baseline: QuranWordAlignmentResult,
        candidate: QuranWordAlignmentResult,
        collapsed: TranscriptWord,
    ) -> bool:
        matched_gain = (
            candidate.matched_word_count
            - baseline.matched_word_count
        )

        if matched_gain < self.minimum_match_gain:
            return False

        if not self._adds_anchored_repeat(
            baseline,
            candidate,
            collapsed,
        ):
            return False

        if (
            candidate.recognised_coverage
            <
            baseline.recognised_coverage
            - self.maximum_coverage_drop
        ):
            return False

        if (
            candidate.average_confidence
            <
            baseline.average_confidence
            - self.maximum_confidence_drop
        ):
            return False

        if not self._preserves_existing_alignment(
            baseline,
            candidate,
            collapsed,
        ):
            return False

        return True

    def _recover_region_words(
        self,
        audio_path: Path,
        collapsed: TranscriptWord,
    ) -> tuple[TranscriptWord, ...]:
        region_start = max(
            0.0,
            float(collapsed.start)
            + self.recovery_start_offset,
        )

        region_end = (
            float(collapsed.end)
            + self.recovery_end_padding
        )

        try:
            transcription = self.recognizer.transcribe_region(
                audio_path,
                region_start,
                region_end,
            )
        except Exception:
            # Recovery is supplementary. Failure must never break
            # the primary Quran alignment pipeline.
            return ()

        recovered: list[TranscriptWord] = []

        for word in self._flatten_words(transcription):
            word_start = float(word.start)

            if (
                word_start
                <= float(collapsed.start)
            ):
                continue

            if (
                word_start
                >= float(collapsed.end)
            ):
                continue

            if (
                float(word.probability)
                < self.minimum_recovery_probability
            ):
                continue

            # Critical safety boundary:
            # recovered speech may never extend past the original
            # collapsed interval.
            safe_end = min(
                float(word.end),
                float(collapsed.end),
            )

            if safe_end <= word_start:
                continue

            recovered.append(
                replace(
                    word,
                    start=word_start,
                    end=safe_end,
                )
            )

        return tuple(recovered)

    def _build_candidate_words(
        self,
        current_words: Sequence[TranscriptWord],
        collapsed: TranscriptWord,
        recovered_words: Sequence[TranscriptWord],
    ) -> tuple[TranscriptWord, ...]:
        if not recovered_words:
            return tuple(current_words)

        recovery_start = min(
            float(word.start)
            for word in recovered_words
        )

        candidate: list[TranscriptWord] = []

        collapsed_replaced = False

        for word in current_words:
            is_collapsed = (
                not collapsed_replaced
                and word is collapsed
            )

            if is_collapsed:
                if recovery_start <= float(word.start):
                    return tuple(current_words)

                candidate.append(
                    replace(
                        word,
                        end=recovery_start,
                    )
                )

                collapsed_replaced = True
                continue

            duration = self._duration(word)

            # Remove only extremely short, low-confidence fragments
            # sitting at the collapsed boundary.
            #
            # This targets artifacts such as the 0.100 second
            # "بصيرا" fragment found in Test Audio 2 while protecting
            # genuine neighbouring Quran words.
            if (
                float(word.start)
                >= recovery_start
                and
                float(word.start)
                >= float(collapsed.end)
                - self.preserve_time_tolerance
                and
                float(word.end)
                <= float(collapsed.end)
                + self.preserve_time_tolerance
                and
                duration
                <= self.tiny_fragment_duration
                and
                float(word.probability)
                < self.tiny_fragment_max_probability
            ):
                continue

            candidate.append(word)

        if not collapsed_replaced:
            return tuple(current_words)

        candidate.extend(recovered_words)

        candidate.sort(
            key=lambda word: (
                float(word.start),
                float(word.end),
            )
        )

        return tuple(candidate)

    def recover(
        self,
        *,
        audio_path: Path,
        recognised_words: Sequence[TranscriptWord],
        verified_text: str,
        baseline_alignment: QuranWordAlignmentResult | None = None,
    ) -> RepetitionRecoveryResult:
        current_words = tuple(recognised_words)

        if baseline_alignment is None:
            baseline_alignment = self.word_aligner.align(
                current_words,
                verified_text,
            )

        if baseline_alignment is None:
            return RepetitionRecoveryResult(
                recognised_words=current_words,
                word_alignment=None,
                attempted_regions=0,
                accepted_regions=0,
            )

        current_alignment = baseline_alignment

        suspicious_words = tuple(
            word
            for word in current_words
            if self._duration(word)
            > self.suspicious_duration
        )

        attempted_regions = 0
        accepted_regions = 0

        for collapsed in suspicious_words:
            attempted_regions += 1

            recovered_words = self._recover_region_words(
                audio_path,
                collapsed,
            )

            if not recovered_words:
                continue

            candidate_words = self._build_candidate_words(
                current_words,
                collapsed,
                recovered_words,
            )

            if candidate_words == current_words:
                continue

            candidate_alignment = self.word_aligner.align(
                candidate_words,
                verified_text,
            )

            if candidate_alignment is None:
                continue

            if not self._accept_candidate(
                current_alignment,
                candidate_alignment,
                collapsed,
            ):
                continue

            current_words = candidate_words
            current_alignment = candidate_alignment
            accepted_regions += 1

        return RepetitionRecoveryResult(
            recognised_words=current_words,
            word_alignment=current_alignment,
            attempted_regions=attempted_regions,
            accepted_regions=accepted_regions,
        )
