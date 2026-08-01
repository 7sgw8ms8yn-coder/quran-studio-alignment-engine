from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.services.quran_candidate_matcher import QuranCandidateMatcher
from app.services.quran_corpus_service import QuranCorpus
from app.services.speech_recognition_service import (
    SpeechRecognitionError,
    TranscriptionResult,
)


class QuranAlignmentError(RuntimeError):
    """Raised when Quran audio cannot be aligned safely."""


class SpeechRecognizerProtocol(Protocol):
    def transcribe(
        self,
        audio_path: Path,
    ) -> TranscriptionResult:
        ...


@dataclass(frozen=True, slots=True)
class AlignedAyah:
    surah_number: int
    ayah_number: int
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class QuranAlignmentResult:
    transcript: str
    surah_number: int
    start_ayah: int
    end_ayah: int
    match_score: float
    duration: float
    ayahs: tuple[AlignedAyah, ...]

    @property
    def verified_text(self) -> str:
        return " ".join(
            ayah.text
            for ayah in self.ayahs
        ).strip()


class QuranAlignmentService:
    """
    Connects rough speech recognition to verified Quran text.

    The recognizer is used only to locate likely ayahs. Final Arabic text
    always comes from the verified Quran corpus.
    """

    def __init__(
        self,
        corpus: QuranCorpus,
        recognizer: SpeechRecognizerProtocol,
        max_sequence_length: int = 4,
        minimum_match_score: float = 0.58,
    ) -> None:
        if not 0.0 <= minimum_match_score <= 1.0:
            raise ValueError(
                "minimum_match_score must be between 0 and 1."
            )

        self.corpus = corpus
        self.recognizer = recognizer
        self.minimum_match_score = minimum_match_score
        self.matcher = QuranCandidateMatcher(
            corpus=corpus,
            max_sequence_length=max_sequence_length,
        )

    def align(
        self,
        audio_path: Path,
    ) -> QuranAlignmentResult:
        try:
            transcription = self.recognizer.transcribe(
                audio_path
            )
        except SpeechRecognitionError as error:
            raise QuranAlignmentError(
                "The audio could not be transcribed."
            ) from error

        transcript = transcription.text.strip()

        if not transcript:
            raise QuranAlignmentError(
                "The speech recognizer returned no Arabic text."
            )

        raw_candidates = self.matcher.find_candidates(
            transcript,
            limit=2,
        )

        candidate = self.matcher.best_match(
            transcript,
            minimum_score=self.minimum_match_score,
        )

        if candidate is None:
            diagnostic_parts = [
                "No sufficiently reliable Quran match was found.",
                f"Transcript: {transcript!r}",
                f"Required score: {self.minimum_match_score:.3f}",
            ]

            for index, raw_candidate in enumerate(raw_candidates, start=1):
                diagnostic_parts.append(
                    f"Candidate {index}: "
                    f"surah={raw_candidate.surah_number}, "
                    f"ayahs={raw_candidate.start_ayah}-{raw_candidate.end_ayah}, "
                    f"score={raw_candidate.score:.3f}, "
                    f"fuzzy={raw_candidate.fuzzy_coverage:.3f}, "
                    f"ngram={raw_candidate.ngram_similarity:.3f}"
                )

            if len(raw_candidates) > 1:
                diagnostic_parts.append(
                    "Score margin: "
                    f"{raw_candidates[0].score - raw_candidates[1].score:.3f}"
                )

            raise QuranAlignmentError(" | ".join(diagnostic_parts))

        verified_ayahs = tuple(
            self.corpus.get_ayah(
                candidate.surah_number,
                ayah_number,
            )
            for ayah_number in range(
                candidate.start_ayah,
                candidate.end_ayah + 1,
            )
        )

        duration = max(
            0.0,
            float(transcription.duration),
        )

        word_counts = [
            max(1, len(ayah.text.split()))
            for ayah in verified_ayahs
        ]

        total_words = sum(word_counts)
        aligned_ayahs: list[AlignedAyah] = []
        elapsed = 0.0

        for index, (ayah, word_count) in enumerate(
            zip(
                verified_ayahs,
                word_counts,
                strict=True,
            )
        ):
            start = elapsed

            if index == len(verified_ayahs) - 1:
                end = duration
            else:
                share = word_count / total_words
                end = elapsed + duration * share

            aligned_ayahs.append(
                AlignedAyah(
                    surah_number=ayah.surah_number,
                    ayah_number=ayah.ayah_number,
                    text=ayah.text,
                    start=round(start, 3),
                    end=round(end, 3),
                )
            )

            elapsed = end

        return QuranAlignmentResult(
            transcript=transcript,
            surah_number=candidate.surah_number,
            start_ayah=candidate.start_ayah,
            end_ayah=candidate.end_ayah,
            match_score=candidate.score,
            duration=duration,
            ayahs=tuple(aligned_ayahs),
        )
