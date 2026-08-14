from pathlib import Path

from app.services.quran_repetition_recovery_service import (
    QuranRepetitionRecoveryService,
)
from app.services.quran_word_alignment_service import (
    QuranWordAlignmentResult,
    TimedVerifiedWord,
)
from app.services.speech_recognition_service import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)


def transcript_word(
    text: str,
    start: float,
    end: float,
    probability: float = 0.99,
) -> TranscriptWord:
    return TranscriptWord(
        text=text,
        start=start,
        end=end,
        probability=probability,
    )


def aligned_word(
    index: int,
    text: str,
    start: float,
    end: float,
) -> TimedVerifiedWord:
    return TimedVerifiedWord(
        verified_index=index,
        text=text,
        start=start,
        end=end,
        confidence=0.99,
        recognised_text=text,
    )


def alignment(
    words,
    recognised_count: int,
    matched_count: int,
    coverage: float,
    confidence: float = 0.99,
) -> QuranWordAlignmentResult:
    return QuranWordAlignmentResult(
        words=tuple(words),
        verified_word_count=40,
        recognised_word_count=recognised_count,
        matched_word_count=matched_count,
        recognised_coverage=coverage,
        verified_coverage=(
            matched_count / 40
        ),
        average_confidence=confidence,
    )


class FakeRecognizer:
    def __init__(self, words):
        self.words = tuple(words)

    def transcribe_region(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            language="ar",
            language_probability=1.0,
            duration=end_seconds - start_seconds,
            segments=(
                TranscriptSegment(
                    text=" ".join(
                        word.text
                        for word in self.words
                    ),
                    start=start_seconds,
                    end=end_seconds,
                    words=self.words,
                ),
            ),
        )


class FakeAligner:
    def __init__(self, candidate):
        self.candidate = candidate

    def align(self, recognised_words, verified_text):
        return self.candidate


def test_no_long_word_does_not_attempt_recovery():
    words = (
        transcript_word(
            "وكان",
            0.0,
            1.0,
        ),
        transcript_word(
            "أبوهما",
            1.0,
            2.0,
        ),
    )

    baseline = alignment(
        (
            aligned_word(
                21,
                "وكان",
                0.0,
                1.0,
            ),
            aligned_word(
                22,
                "أبوهما",
                1.0,
                2.0,
            ),
        ),
        recognised_count=2,
        matched_count=2,
        coverage=1.0,
    )

    service = QuranRepetitionRecoveryService(
        recognizer=FakeRecognizer(()),
        word_aligner=FakeAligner(baseline),
    )

    result = service.recover(
        audio_path=Path("unused.wav"),
        recognised_words=words,
        verified_text="وكان أبوهما",
        baseline_alignment=baseline,
    )

    assert result.attempted_regions == 0
    assert result.accepted_regions == 0
    assert result.recognised_words == words
    assert result.word_alignment == baseline


def test_boundary_safe_recovery_is_accepted():
    collapsed = transcript_word(
        "صالحا",
        23.4,
        29.88,
    )

    original = (
        transcript_word(
            "وكان",
            20.8,
            21.84,
        ),
        transcript_word(
            "أبوهما",
            21.84,
            23.4,
        ),
        collapsed,
        transcript_word(
            "فأراد",
            29.98,
            31.38,
        ),
    )

    baseline = alignment(
        (
            aligned_word(
                21,
                "وكان",
                20.8,
                21.84,
            ),
            aligned_word(
                22,
                "أبوهما",
                21.84,
                23.4,
            ),
            aligned_word(
                23,
                "صالحا",
                23.4,
                29.88,
            ),
            aligned_word(
                24,
                "فأراد",
                29.98,
                31.38,
            ),
        ),
        recognised_count=4,
        matched_count=4,
        coverage=1.0,
    )

    candidate = alignment(
        (
            aligned_word(
                21,
                "وكان",
                20.8,
                21.84,
            ),
            aligned_word(
                22,
                "أبوهما",
                21.84,
                23.4,
            ),
            aligned_word(
                23,
                "صالحا",
                23.4,
                25.34,
            ),
            aligned_word(
                21,
                "وكان",
                25.34,
                26.74,
            ),
            aligned_word(
                22,
                "أبوهما",
                26.74,
                28.14,
            ),
            aligned_word(
                23,
                "صالحا",
                28.14,
                29.88,
            ),
            aligned_word(
                24,
                "فأراد",
                29.98,
                31.38,
            ),
        ),
        recognised_count=7,
        matched_count=7,
        coverage=1.0,
    )

    recovery_words = (
        transcript_word(
            "وكان",
            25.34,
            26.74,
        ),
        transcript_word(
            "أبوهما",
            26.74,
            28.14,
        ),
        transcript_word(
            "صالحا",
            28.14,
            30.10,
        ),
    )

    service = QuranRepetitionRecoveryService(
        recognizer=FakeRecognizer(
            recovery_words
        ),
        word_aligner=FakeAligner(
            candidate
        ),
    )

    result = service.recover(
        audio_path=Path("unused.wav"),
        recognised_words=original,
        verified_text=(
            "وكان أبوهما صالحا "
            "وكان أبوهما صالحا فأراد"
        ),
        baseline_alignment=baseline,
    )

    assert result.attempted_regions == 1
    assert result.accepted_regions == 1
    assert result.word_alignment == candidate

    recovered_last = [
        word
        for word in result.recognised_words
        if word.text == "صالحا"
        and word.start >= 28.0
    ][0]

    assert recovered_last.end == 29.88

    following = [
        word
        for word in result.recognised_words
        if word.text == "فأراد"
    ]

    assert len(following) == 1
    assert following[0].start == 29.98


def test_recovery_rejected_if_following_word_disappears():
    collapsed = transcript_word(
        "صالحا",
        23.4,
        29.88,
    )

    original = (
        transcript_word(
            "وكان",
            20.8,
            21.84,
        ),
        collapsed,
        transcript_word(
            "فأراد",
            29.98,
            31.38,
        ),
    )

    baseline = alignment(
        (
            aligned_word(
                21,
                "وكان",
                20.8,
                21.84,
            ),
            aligned_word(
                23,
                "صالحا",
                23.4,
                29.88,
            ),
            aligned_word(
                24,
                "فأراد",
                29.98,
                31.38,
            ),
        ),
        recognised_count=3,
        matched_count=3,
        coverage=1.0,
    )

    # Candidate gains repeated Quran words but loses
    # verified word 24 "فأراد".
    unsafe_candidate = alignment(
        (
            aligned_word(
                21,
                "وكان",
                20.8,
                21.84,
            ),
            aligned_word(
                23,
                "صالحا",
                23.4,
                25.34,
            ),
            aligned_word(
                21,
                "وكان",
                25.34,
                26.74,
            ),
            aligned_word(
                22,
                "أبوهما",
                26.74,
                28.14,
            ),
            aligned_word(
                23,
                "صالحا",
                28.14,
                29.88,
            ),
        ),
        recognised_count=6,
        matched_count=5,
        coverage=0.833333,
    )

    service = QuranRepetitionRecoveryService(
        recognizer=FakeRecognizer(
            (
                transcript_word(
                    "وكان",
                    25.34,
                    26.74,
                ),
                transcript_word(
                    "أبوهما",
                    26.74,
                    28.14,
                ),
                transcript_word(
                    "صالحا",
                    28.14,
                    30.10,
                ),
            )
        ),
        word_aligner=FakeAligner(
            unsafe_candidate
        ),
        # Allow coverage drop here so this test specifically
        # proves that baseline-word preservation rejects it.
        maximum_coverage_drop=1.0,
    )

    result = service.recover(
        audio_path=Path("unused.wav"),
        recognised_words=original,
        verified_text=(
            "وكان أبوهما صالحا "
            "وكان أبوهما صالحا فأراد"
        ),
        baseline_alignment=baseline,
    )

    assert result.attempted_regions == 1
    assert result.accepted_regions == 0
    assert result.recognised_words == original
    assert result.word_alignment == baseline


def test_recovery_rejected_without_anchor_repeat():
    collapsed = transcript_word(
        "صالحا",
        23.4,
        29.88,
    )

    original = (
        transcript_word(
            "وكان",
            20.8,
            21.84,
        ),
        collapsed,
        transcript_word(
            "فأراد",
            29.98,
            31.38,
        ),
    )

    baseline = alignment(
        (
            aligned_word(
                21,
                "وكان",
                20.8,
                21.84,
            ),
            aligned_word(
                23,
                "صالحا",
                23.4,
                29.88,
            ),
            aligned_word(
                24,
                "فأراد",
                29.98,
                31.38,
            ),
        ),
        recognised_count=3,
        matched_count=3,
        coverage=1.0,
    )

    # This candidate appears numerically better and preserves
    # the following Quran word, but it does NOT add another
    # occurrence of the collapsed Quran anchor (index 23).
    unsafe_candidate = alignment(
        (
            aligned_word(
                21,
                "وكان",
                20.8,
                21.84,
            ),
            aligned_word(
                23,
                "صالحا",
                23.4,
                25.34,
            ),
            aligned_word(
                22,
                "أبوهما",
                25.34,
                26.74,
            ),
            aligned_word(
                24,
                "فأراد",
                29.98,
                31.38,
            ),
        ),
        recognised_count=4,
        matched_count=4,
        coverage=1.0,
    )

    service = QuranRepetitionRecoveryService(
        recognizer=FakeRecognizer(
            (
                transcript_word(
                    "أبوهما",
                    25.34,
                    26.74,
                ),
            )
        ),
        word_aligner=FakeAligner(
            unsafe_candidate
        ),
        maximum_coverage_drop=1.0,
    )

    result = service.recover(
        audio_path=Path("unused.wav"),
        recognised_words=original,
        verified_text=(
            "وكان أبوهما صالحا فأراد"
        ),
        baseline_alignment=baseline,
    )

    assert result.attempted_regions == 1
    assert result.accepted_regions == 0
    assert result.recognised_words == original
    assert result.word_alignment == baseline
