from pathlib import Path

import pytest

from app.services.quran_alignment_service import (
    QuranAlignmentError,
    QuranAlignmentService,
)
from app.services.quran_corpus_service import load_quran_corpus
from app.services.speech_recognition_service import (
    SpeechRecognitionError,
    TranscriptSegment,
    TranscriptionResult,
)


class FakeRecognizer:
    def __init__(
        self,
        result: TranscriptionResult,
    ) -> None:
        self.result = result

    def transcribe(
        self,
        audio_path: Path,
    ) -> TranscriptionResult:
        return self.result


class FailingRecognizer:
    def transcribe(
        self,
        audio_path: Path,
    ) -> TranscriptionResult:
        raise SpeechRecognitionError(
            "Simulated transcription failure."
        )


def make_transcription(
    text: str,
    duration: float = 10.0,
) -> TranscriptionResult:
    return TranscriptionResult(
        language="ar",
        language_probability=1.0,
        duration=duration,
        segments=(
            TranscriptSegment(
                text=text,
                start=0.0,
                end=duration,
                words=(),
            ),
        ),
    )


def test_alignment_uses_verified_quran_text() -> None:
    corpus = load_quran_corpus()
    recognizer = FakeRecognizer(
        make_transcription(
            "الحمد لله رب العلمين"
        )
    )

    service = QuranAlignmentService(
        corpus=corpus,
        recognizer=recognizer,
    )

    result = service.align(
        Path("unused-test-audio.mp3")
    )

    assert result.surah_number == 1
    assert result.start_ayah == 2
    assert result.end_ayah == 2
    assert result.ayahs[0].text == (
        corpus.get_ayah(1, 2).text
    )
    assert result.verified_text == (
        corpus.get_ayah(1, 2).text
    )
    assert result.match_score > 0.70


def test_consecutive_ayahs_receive_timestamps() -> None:
    corpus = load_quran_corpus()
    recognizer = FakeRecognizer(
        make_transcription(
            "الحمد لله رب العالمين الرحمن الرحيم",
            duration=12.0,
        )
    )

    service = QuranAlignmentService(
        corpus=corpus,
        recognizer=recognizer,
        max_sequence_length=3,
    )

    result = service.align(
        Path("unused-test-audio.mp3")
    )

    assert result.surah_number == 1
    assert result.start_ayah == 2
    assert result.end_ayah == 3
    assert len(result.ayahs) == 2

    first, second = result.ayahs

    assert first.start == 0.0
    assert first.end > first.start
    assert second.start == first.end
    assert second.end == 12.0


def test_empty_transcript_is_rejected() -> None:
    corpus = load_quran_corpus()
    recognizer = FakeRecognizer(
        make_transcription("")
    )

    service = QuranAlignmentService(
        corpus=corpus,
        recognizer=recognizer,
    )

    with pytest.raises(
        QuranAlignmentError,
        match="returned no Arabic text",
    ):
        service.align(
            Path("unused-test-audio.mp3")
        )


def test_low_quality_match_is_rejected() -> None:
    corpus = load_quran_corpus()
    recognizer = FakeRecognizer(
        make_transcription(
            "كلام غير مطابق للنص المطلوب تماما"
        )
    )

    service = QuranAlignmentService(
        corpus=corpus,
        recognizer=recognizer,
        minimum_match_score=0.90,
    )

    with pytest.raises(
        QuranAlignmentError,
        match="No sufficiently reliable",
    ):
        service.align(
            Path("unused-test-audio.mp3")
        )


def test_recognition_failure_is_wrapped() -> None:
    corpus = load_quran_corpus()

    service = QuranAlignmentService(
        corpus=corpus,
        recognizer=FailingRecognizer(),
    )

    with pytest.raises(
        QuranAlignmentError,
        match="could not be transcribed",
    ):
        service.align(
            Path("unused-test-audio.mp3")
        )
