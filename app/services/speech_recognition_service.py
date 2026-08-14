from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


class SpeechRecognitionError(RuntimeError):
    """Raised when the speech-recognition model or transcription fails."""


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: tuple[TranscriptWord, ...]


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    segments: tuple[TranscriptSegment, ...]

    @property
    def text(self) -> str:
        return " ".join(
            segment.text.strip()
            for segment in self.segments
            if segment.text.strip()
        ).strip()


class ArabicSpeechRecognizer:
    """
    Lazily loads Faster-Whisper for Arabic recitation transcription.

    Model loading is deliberately deferred until the first transcription
    request so the API can start without consuming model memory.
    """

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        self._model: WhisperModel | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                self._load_error = None
            except Exception as error:
                self._model = None
                self._load_error = str(error)

                raise SpeechRecognitionError(
                    "The speech-recognition model could not be loaded."
                ) from error

    def transcribe(
        self,
        audio_path: Path,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            raise SpeechRecognitionError(
                "The prepared audio file does not exist."
            )

        if audio_path.stat().st_size == 0:
            raise SpeechRecognitionError(
                "The prepared audio file is empty."
            )

        self.load()

        if self._model is None:
            raise SpeechRecognitionError(
                "The speech-recognition model is unavailable."
            )

        try:
            raw_segments, info = self._model.transcribe(
                str(audio_path),
                language="ar",
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False,
            )

            converted_segments: list[TranscriptSegment] = []

            for raw_segment in raw_segments:
                converted_words: list[TranscriptWord] = []

                for raw_word in raw_segment.words or ():
                    converted_words.append(
                        TranscriptWord(
                            text=raw_word.word.strip(),
                            start=float(raw_word.start),
                            end=float(raw_word.end),
                            probability=float(raw_word.probability),
                        )
                    )

                converted_segments.append(
                    TranscriptSegment(
                        text=raw_segment.text.strip(),
                        start=float(raw_segment.start),
                        end=float(raw_segment.end),
                        words=tuple(converted_words),
                    )
                )

            language = str(
                getattr(info, "language", "ar")
            )

            language_probability = float(
                getattr(info, "language_probability", 0.0)
            )

            duration = float(
                getattr(info, "duration", 0.0)
            )

            return TranscriptionResult(
                language=language,
                language_probability=language_probability,
                duration=duration,
                segments=tuple(converted_segments),
            )
        except SpeechRecognitionError:
            raise
        except Exception as error:
            raise SpeechRecognitionError(
                "The Arabic audio could not be transcribed."
            ) from error


    def transcribe_region(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> TranscriptionResult:
        """
        Re-transcribe a small region of an audio file.

        This is intended for targeted recovery of suspicious collapsed
        word-timestamp regions without re-running the whole recording.
        """
        if not audio_path.exists():
            raise SpeechRecognitionError(
                "The prepared audio file does not exist."
            )

        if audio_path.stat().st_size == 0:
            raise SpeechRecognitionError(
                "The prepared audio file is empty."
            )

        start_seconds = float(start_seconds)
        end_seconds = float(end_seconds)

        if start_seconds < 0.0:
            raise SpeechRecognitionError(
                "Region start time cannot be negative."
            )

        if end_seconds <= start_seconds:
            raise SpeechRecognitionError(
                "Region end time must be after its start time."
            )

        self.load()

        if self._model is None:
            raise SpeechRecognitionError(
                "The speech-recognition model is unavailable."
            )

        try:
            raw_segments, info = self._model.transcribe(
                str(audio_path),
                language="ar",
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                word_timestamps=True,
                condition_on_previous_text=False,
                clip_timestamps=[
                    start_seconds,
                    end_seconds,
                ],
            )

            converted_segments: list[TranscriptSegment] = []

            for raw_segment in raw_segments:
                converted_words: list[TranscriptWord] = []

                for raw_word in raw_segment.words or ():
                    converted_words.append(
                        TranscriptWord(
                            text=raw_word.word.strip(),
                            start=float(raw_word.start),
                            end=float(raw_word.end),
                            probability=float(
                                raw_word.probability
                            ),
                        )
                    )

                converted_segments.append(
                    TranscriptSegment(
                        text=raw_segment.text.strip(),
                        start=float(raw_segment.start),
                        end=float(raw_segment.end),
                        words=tuple(converted_words),
                    )
                )

            language = str(
                getattr(info, "language", "ar")
            )

            language_probability = float(
                getattr(
                    info,
                    "language_probability",
                    0.0,
                )
            )

            return TranscriptionResult(
                language=language,
                language_probability=language_probability,
                duration=max(
                    0.0,
                    end_seconds - start_seconds,
                ),
                segments=tuple(converted_segments),
            )

        except SpeechRecognitionError:
            raise
        except Exception as error:
            raise SpeechRecognitionError(
                "The Arabic audio region could not be transcribed."
            ) from error


speech_recognizer = ArabicSpeechRecognizer()
