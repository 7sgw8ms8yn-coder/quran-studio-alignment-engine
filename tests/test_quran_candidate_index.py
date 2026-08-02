import time

from app.services.quran_candidate_index import (
    QuranCandidateIndex,
)
from app.services.quran_corpus_service import (
    load_quran_corpus,
)


def test_index_finds_exact_ayah() -> None:
    corpus = load_quran_corpus()
    index = QuranCandidateIndex(corpus)

    results = index.shortlist(
        "الحمد لله رب العالمين",
        limit=10,
    )

    assert results
    assert any(
        result.surah_number == 1
        and result.ayah_number == 2
        for result in results[:5]
    )


def test_index_tolerates_small_spelling_error() -> None:
    corpus = load_quran_corpus()
    index = QuranCandidateIndex(corpus)

    results = index.shortlist(
        "الحمد لله رب العلمين",
        limit=10,
    )

    assert results
    assert any(
        result.surah_number == 1
        and result.ayah_number == 2
        for result in results[:5]
    )


def test_index_expands_nearby_passages() -> None:
    corpus = load_quran_corpus()
    index = QuranCandidateIndex(corpus)

    results = index.shortlist(
        "الحمد لله رب العالمين",
        limit=5,
    )

    references = index.expand_passages(
        results,
        before=1,
        after=1,
    )

    assert (1, 1) in references
    assert (1, 2) in references
    assert (1, 3) in references


def test_shortlist_is_fast() -> None:
    corpus = load_quran_corpus()
    index = QuranCandidateIndex(corpus)

    started = time.perf_counter()

    results = index.shortlist(
        "الحمد لله رب العالمين الرحمن الرحيم",
        limit=40,
    )

    elapsed = (
        time.perf_counter() - started
    )

    assert results
    assert elapsed < 1.0
