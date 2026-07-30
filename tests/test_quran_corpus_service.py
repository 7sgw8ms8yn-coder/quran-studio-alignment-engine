from pathlib import Path

import pytest

from app.services.quran_corpus_service import (
    EXPECTED_AYAH_COUNT,
    QuranCorpusError,
    load_quran_corpus,
    normalize_arabic_for_search,
)


def test_arabic_normalisation() -> None:
    assert (
        normalize_arabic_for_search(
            "إِنَّا أَعْطَيْنَاكَ الْكَوْثَرَ"
        )
        == "انا اعطيناك الكوثر"
    )


def test_missing_corpus_is_rejected(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing-quran.txt"

    with pytest.raises(
        QuranCorpusError,
        match="missing",
    ):
        load_quran_corpus(missing_path)


def test_real_corpus_loads() -> None:
    corpus = load_quran_corpus()

    assert len(corpus.ayahs) == EXPECTED_AYAH_COUNT
    assert corpus.ayahs[0].surah_number == 1
    assert corpus.ayahs[0].ayah_number == 1
    assert corpus.ayahs[-1].surah_number == 114
    assert corpus.ayahs[-1].ayah_number == 6


def test_surah_al_fatiha_has_seven_ayahs() -> None:
    corpus = load_quran_corpus()

    al_fatiha = corpus.get_surah(1)

    assert len(al_fatiha) == 7


def test_exact_search_finds_ayah() -> None:
    corpus = load_quran_corpus()

    results = corpus.search_exact(
        "الحمد لله رب العالمين"
    )

    assert len(results) >= 1
    assert results[0].surah_number == 1
    assert results[0].ayah_number == 2