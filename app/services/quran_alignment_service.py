from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.services.quran_candidate_matcher import QuranCandidateMatcher
from app.services.quran_corpus_service import QuranCorpus
from app.services.quran_word_alignment_service import (
    QuranWordAlignmentResult,
    QuranWordAlignmentService,
)
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


class QuranVerseProviderProtocol(Protocol):
    def get_verse(
        self,
        surah_number: int,
        ayah_number: int,
        *,
        include_words: bool = True,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class AlignedAyah:
    surah_number: int
    ayah_number: int
    text: str
    start: float
    end: float
    text_imlaei: str | None = None
    words: tuple[dict[str, Any], ...] = ()
    verification_source: str = "local_corpus"


@dataclass(frozen=True, slots=True)
class QuranAlignmentResult:
    transcript: str
    surah_number: int
    start_ayah: int
    end_ayah: int
    match_score: float
    duration: float
    ayahs: tuple[AlignedAyah, ...]
    word_alignment: QuranWordAlignmentResult | None = None

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
        verse_provider: QuranVerseProviderProtocol | None = None,
        word_aligner: QuranWordAlignmentService | None = None,
    ) -> None:
        if not 0.0 <= minimum_match_score <= 1.0:
            raise ValueError(
                "minimum_match_score must be between 0 and 1."
            )

        self.corpus = corpus
        self.recognizer = recognizer
        self.minimum_match_score = minimum_match_score
        self.verse_provider = verse_provider
        self.word_aligner = (
            word_aligner
            if word_aligner is not None
            else QuranWordAlignmentService()
        )
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

        local_ayahs = tuple(
            self.corpus.get_ayah(
                candidate.surah_number,
                ayah_number,
            )
            for ayah_number in range(
                candidate.start_ayah,
                candidate.end_ayah + 1,
            )
        )

        verified_entries: list[
            tuple[
                Any,
                str,
                str | None,
                tuple[dict[str, Any], ...],
                str,
            ]
        ] = []

        for local_ayah in local_ayahs:
            if self.verse_provider is None:
                verified_entries.append(
                    (
                        local_ayah,
                        local_ayah.text,
                        None,
                        (),
                        "local_corpus",
                    )
                )
                continue

            try:
                remote_verse = self.verse_provider.get_verse(
                    local_ayah.surah_number,
                    local_ayah.ayah_number,
                    include_words=True,
                )
            except Exception as error:
                raise QuranAlignmentError(
                    "The detected Quran passage could not be "
                    "verified with Quran Foundation."
                ) from error

            verified_text = (
                remote_verse.text_uthmani
                or remote_verse.text_imlaei
            )

            if not verified_text:
                raise QuranAlignmentError(
                    "Quran Foundation returned no verified Arabic "
                    f"for {local_ayah.surah_number}:"
                    f"{local_ayah.ayah_number}."
                )

            verified_entries.append(
                (
                    local_ayah,
                    verified_text,
                    remote_verse.text_imlaei,
                    tuple(remote_verse.words),
                    "quran_foundation",
                )
            )

        duration = max(
            0.0,
            float(transcription.duration),
        )

        recognised_words = tuple(
            word
            for segment in transcription.segments
            for word in segment.words
        )

        word_alignment: QuranWordAlignmentResult | None = None

        verified_word_text = " ".join(
            (
                text_imlaei
                if text_imlaei
                else verified_text
            )
            for (
                _,
                verified_text,
                text_imlaei,
                _,
                _,
            ) in verified_entries
        ).strip()

        if recognised_words and verified_word_text:
            word_alignment = self.word_aligner.align(
                recognised_words,
                verified_word_text,
            )

        word_counts = [
            max(1, len(text.split()))
            for _, text, _, _, _ in verified_entries
        ]

        total_words = sum(word_counts)
        aligned_ayahs: list[AlignedAyah] = []
        elapsed = 0.0

        for index, (entry, word_count) in enumerate(
            zip(
                verified_entries,
                word_counts,
                strict=True,
            )
        ):
            (
                ayah,
                verified_text,
                text_imlaei,
                words,
                verification_source,
            ) = entry
            start = elapsed

            if index == len(verified_entries) - 1:
                end = duration
            else:
                share = word_count / total_words
                end = elapsed + duration * share

            aligned_ayahs.append(
                AlignedAyah(
                    surah_number=ayah.surah_number,
                    ayah_number=ayah.ayah_number,
                    text=verified_text,
                    start=round(start, 3),
                    end=round(end, 3),
                    text_imlaei=text_imlaei,
                    words=words,
                    verification_source=verification_source,
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
            word_alignment=word_alignment,
        )
