from app.services.quran_candidate_matcher import (
    QuranCandidateMatcher,
    normalise_for_matching,
)
from app.services.quran_corpus_service import (
    load_quran_corpus,
)


def test_matching_normalisation() -> None:
    assert normalise_for_matching(
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
    ) == "الحمد لله رب العالمين"


def test_exact_ayah_match() -> None:
    corpus = load_quran_corpus()
    matcher = QuranCandidateMatcher(corpus)

    match = matcher.best_match(
        "الحمد لله رب العالمين"
    )

    assert match is not None
    assert match.surah_number == 1
    assert match.start_ayah == 2
    assert match.end_ayah == 2
    assert match.score > 0.90
    assert match.word_overlap > 0.90
    assert match.transcript_coverage > 0.90


def test_imperfect_transcript_finds_correct_ayah() -> None:
    corpus = load_quran_corpus()
    matcher = QuranCandidateMatcher(corpus)

    match = matcher.best_match(
        "الحمد لله رب العلمين"
    )

    assert match is not None
    assert match.surah_number == 1
    assert match.start_ayah == 2
    assert match.score > 0.70
    assert match.transcript_coverage >= 0.75


def test_consecutive_ayah_sequence() -> None:
    corpus = load_quran_corpus()

    matcher = QuranCandidateMatcher(
        corpus,
        max_sequence_length=3,
    )

    match = matcher.best_match(
        "الحمد لله رب العالمين الرحمن الرحيم"
    )

    assert match is not None
    assert match.surah_number == 1
    assert match.start_ayah == 2
    assert match.end_ayah == 3
    assert match.score > 0.85


def test_very_short_transcript_returns_no_match() -> None:
    corpus = load_quran_corpus()
    matcher = QuranCandidateMatcher(corpus)

    assert matcher.best_match("ال") is None


def test_unrelated_arabic_is_safely_rejected() -> None:
    corpus = load_quran_corpus()
    matcher = QuranCandidateMatcher(corpus)

    match = matcher.best_match(
        "هذا كلام عشوائي لا يرتبط بالنص المطلوب"
    )

    assert match is None


def test_false_match_with_low_word_overlap_is_rejected() -> None:
    corpus = load_quran_corpus()
    matcher = QuranCandidateMatcher(corpus)

    match = matcher.best_match(
        "ازرور تنمغول غردوف زاروف ملحنين"
    )

    assert match is None


def test_phonetic_normalisation_supports_sound_alikes() -> None:
    from app.services.quran_candidate_matcher import (
        phonetic_normalise,
    )

    assert phonetic_normalise("صراط") == "سرات"
    assert phonetic_normalise("ذلك") == "زلك"


def test_character_fragments_detect_imperfect_text() -> None:
    from app.services.quran_candidate_matcher import (
        character_ngrams,
        dice_similarity,
    )

    first = character_ngrams(
        "الحمد لله رب العالمين"
    )

    second = character_ngrams(
        "الحمد لله رب العلمين"
    )

    assert dice_similarity(first, second) > 0.75
