from pathlib import Path

import pytest

from app.services.speech_recognition_service import (
    ArabicSpeechRecognizer,
    SpeechRecognitionError,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)


def test_recognizer_starts_unloaded() -> None:
    recognizer = ArabicSpeechRecognizer()

    assert recognizer.is_loaded is False
    assert recognizer.load_error is None


def test_missing_audio_is_rejected(
    tmp_path: Path,
) -> None:
    recognizer = ArabicSpeechRecognizer()
    missing_audio = tmp_path / "missing.wav"

    with pytest.raises(
        SpeechRecognitionError,
        match="does not exist",
    ):
        recognizer.transcribe(missing_audio)

    assert recognizer.is_loaded is False


def test_transcription_result_combines_segments() -> None:
    result = TranscriptionResult(
        language="ar",
        language_probability=1.0,
        duration=2.0,
        segments=(
            TranscriptSegment(
                text="بسم الله",
                start=0.0,
                end=1.0,
                words=(
                    TranscriptWord(
                        text="بسم",
                        start=0.0,
                        end=0.5,
                        probability=0.99,
                    ),
                    TranscriptWord(
                        text="الله",
                        start=0.5,
                        end=1.0,
                        probability=0.99,
                    ),
                ),
            ),
            TranscriptSegment(
                text="الرحمن الرحيم",
                start=1.0,
                end=2.0,
                words=(),
            ),
        ),
    )

    assert result.text == "بسم الله الرحمن الرحيم"
