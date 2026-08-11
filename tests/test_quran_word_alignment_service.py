from app.services.quran_word_alignment_service import (
    QuranWordAlignmentService,
)
from app.services.speech_recognition_service import TranscriptWord


def test_extracts_words_and_removes_pause_symbols() -> None:
    service = QuranWordAlignmentService()

    words = service.extract_spoken_words(
        "وَاتَّبَعُوا مَا ۖ تَتْلُو ۚ"
    )

    assert words == (
        "وَاتَّبَعُوا",
        "مَا",
        "تَتْلُو",
    )


def test_aligns_partial_recitation_to_verified_words() -> None:
    service = QuranWordAlignmentService()

    recognised = (
        TranscriptWord(
            text="واتبعوا",
            start=0.0,
            end=1.5,
            probability=0.95,
        ),
        TranscriptWord(
            text="ما",
            start=1.5,
            end=1.9,
            probability=0.98,
        ),
        TranscriptWord(
            text="تتلو",
            start=1.9,
            end=2.4,
            probability=0.96,
        ),
    )

    result = service.align(
        recognised,
        "وَاتَّبَعُوا مَا تَتْلُو الشَّيَاطِينُ",
    )

    assert result.matched_word_count == 3
    assert result.recognised_coverage == 1.0
    assert result.verified_coverage == 0.75

    assert result.words[0].verified_index == 1
    assert result.words[0].start == 0.0
    assert result.words[0].end == 1.5


def test_empty_recognition_returns_empty_result() -> None:
    service = QuranWordAlignmentService()

    result = service.align(
        (),
        "وَاتَّبَعُوا مَا",
    )

    assert result.matched_word_count == 0
    assert result.words == ()


def test_rejects_low_confidence_short_timestamp() -> None:
    service = QuranWordAlignmentService(
        minimum_confidence=0.60,
        minimum_duration=0.08,
    )

    recognised = (
        TranscriptWord(
            text="وما",
            start=22.46,
            end=22.50,
            probability=0.60,
        ),
    )

    result = service.align(
        recognised,
        "وَمَا",
    )

    assert result.matched_word_count == 0
    assert result.words == ()


def test_rejects_timestamp_that_moves_backwards() -> None:
    service = QuranWordAlignmentService(
        minimum_confidence=0.50,
        minimum_duration=0.05,
        timestamp_tolerance=0.01,
    )

    recognised = (
        TranscriptWord(
            text="واتبعوا",
            start=1.0,
            end=2.0,
            probability=1.0,
        ),
        TranscriptWord(
            text="ما",
            start=0.5,
            end=1.2,
            probability=1.0,
        ),
    )

    result = service.align(
        recognised,
        "وَاتَّبَعُوا مَا",
    )

    assert result.matched_word_count == 1
    assert result.words[0].text == "وَاتَّبَعُوا"



def test_recovers_single_final_quran_word_from_fragmented_whisper_tail() -> None:
    service = QuranWordAlignmentService()

    recognised = (
        TranscriptWord(
            text="عَلَيْهِ",
            start=0.0,
            end=1.0,
            probability=0.99,
        ),
        TranscriptWord(
            text="صَادًا",
            start=1.0,
            end=1.14,
            probability=0.6128154590725898,
        ),
        TranscriptWord(
            text="مَصَدًا",
            start=1.14,
            end=2.18,
            probability=0.414850511030973,
        ),
        TranscriptWord(
            text="مَصَدًا",
            start=2.18,
            end=2.66,
            probability=0.3122150605415186,
        ),
    )

    result = service.align(
        recognised,
        "عَلَيْهِ صَبْرًا",
    )

    assert result.matched_word_count == 2

    final_word = result.words[-1]

    assert final_word.verified_index == 2
    assert final_word.text == "صَبْرًا"
    assert final_word.start == 1.0
    assert final_word.end == 2.66

    # Recovery deliberately preserves the weak ASR confidence
    # instead of pretending this was a normal high-confidence match.
    assert final_word.confidence < 0.60


def test_does_not_recover_when_multiple_verified_words_remain() -> None:
    service = QuranWordAlignmentService()

    recognised = (
        TranscriptWord(
            text="عَلَيْهِ",
            start=0.0,
            end=1.0,
            probability=0.99,
        ),
        TranscriptWord(
            text="صَادًا",
            start=1.0,
            end=1.2,
            probability=0.62,
        ),
    )

    result = service.align(
        recognised,
        "عَلَيْهِ كَلِمَة صَبْرًا",
    )

    assert result.matched_word_count == 1
    assert result.words[-1].text == "عَلَيْهِ"
