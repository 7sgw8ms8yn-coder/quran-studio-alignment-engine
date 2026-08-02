from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from app.services.quran_candidate_matcher import (
    character_ngrams,
    normalise_for_matching,
    phonetic_normalise,
)
from app.services.quran_corpus_service import QuranCorpus


@dataclass(frozen=True, slots=True)
class IndexedAyahCandidate:
    surah_number: int
    ayah_number: int
    retrieval_score: float
    token_score: float
    ngram_score: float
    phonetic_score: float


class QuranCandidateIndex:
    """
    Fast first-stage retrieval over individual Quran ayahs.

    This index does not make the final Quran decision. It only narrows the
    corpus to a small shortlist for detailed scoring and acoustic verification.
    """

    def __init__(
        self,
        corpus: QuranCorpus,
        *,
        ngram_size: int = 3,
    ) -> None:
        if ngram_size < 2:
            raise ValueError("ngram_size must be at least 2.")

        self.corpus = corpus
        self.ngram_size = ngram_size

        self._ayahs_by_key = {
            (ayah.surah_number, ayah.ayah_number): ayah
            for ayah in corpus.ayahs
        }

        self._token_postings: dict[
            str,
            set[tuple[int, int]],
        ] = defaultdict(set)

        self._ngram_postings: dict[
            str,
            set[tuple[int, int]],
        ] = defaultdict(set)

        self._phonetic_postings: dict[
            str,
            set[tuple[int, int]],
        ] = defaultdict(set)

        self._normalised_words: dict[
            tuple[int, int],
            set[str],
        ] = {}

        self._normalised_ngrams: dict[
            tuple[int, int],
            set[str],
        ] = {}

        self._phonetic_words: dict[
            tuple[int, int],
            set[str],
        ] = {}

        self._build()

    def _build(self) -> None:
        for ayah in self.corpus.ayahs:
            key = (
                ayah.surah_number,
                ayah.ayah_number,
            )

            normalised = normalise_for_matching(
                ayah.text
            )

            normalised_words = set(
                normalised.split()
            )

            normalised_ngrams = character_ngrams(
                normalised,
                size=self.ngram_size,
            )

            phonetic_words = set(
                phonetic_normalise(
                    normalised
                ).split()
            )

            self._normalised_words[key] = (
                normalised_words
            )

            self._normalised_ngrams[key] = (
                normalised_ngrams
            )

            self._phonetic_words[key] = (
                phonetic_words
            )

            for token in normalised_words:
                self._token_postings[token].add(key)

            for ngram in normalised_ngrams:
                self._ngram_postings[ngram].add(key)

            for token in phonetic_words:
                self._phonetic_postings[token].add(key)

    @staticmethod
    def _dice(
        first: set[str],
        second: set[str],
    ) -> float:
        if not first or not second:
            return 0.0

        return (
            2.0 * len(first & second)
            / (len(first) + len(second))
        )

    def shortlist(
        self,
        transcript: str,
        *,
        limit: int = 40,
    ) -> tuple[IndexedAyahCandidate, ...]:
        normalised = normalise_for_matching(
            transcript
        )

        if len(normalised) < 3:
            return ()

        query_words = set(normalised.split())

        query_ngrams = character_ngrams(
            normalised,
            size=self.ngram_size,
        )

        query_phonetic_words = set(
            phonetic_normalise(
                normalised
            ).split()
        )

        votes: Counter[tuple[int, int]] = Counter()

        for token in query_words:
            for key in self._token_postings.get(
                token,
                (),
            ):
                votes[key] += 5

        for token in query_phonetic_words:
            for key in self._phonetic_postings.get(
                token,
                (),
            ):
                votes[key] += 3

        for ngram in query_ngrams:
            for key in self._ngram_postings.get(
                ngram,
                (),
            ):
                votes[key] += 1

        if not votes:
            return ()

        # Score only the strongest posting-list candidates rather than
        # scanning and deeply comparing all 6,236 ayahs.
        preselected_keys = [
            key
            for key, _ in votes.most_common(
                max(limit * 8, 100)
            )
        ]

        candidates: list[IndexedAyahCandidate] = []

        for key in preselected_keys:
            token_score = self._dice(
                query_words,
                self._normalised_words[key],
            )

            ngram_score = self._dice(
                query_ngrams,
                self._normalised_ngrams[key],
            )

            phonetic_score = self._dice(
                query_phonetic_words,
                self._phonetic_words[key],
            )

            retrieval_score = (
                token_score * 0.35
                + ngram_score * 0.40
                + phonetic_score * 0.25
            )

            candidates.append(
                IndexedAyahCandidate(
                    surah_number=key[0],
                    ayah_number=key[1],
                    retrieval_score=round(
                        retrieval_score,
                        6,
                    ),
                    token_score=round(
                        token_score,
                        6,
                    ),
                    ngram_score=round(
                        ngram_score,
                        6,
                    ),
                    phonetic_score=round(
                        phonetic_score,
                        6,
                    ),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.retrieval_score,
                candidate.ngram_score,
                candidate.phonetic_score,
            ),
            reverse=True,
        )

        safe_limit = max(
            1,
            min(limit, 100),
        )

        return tuple(
            candidates[:safe_limit]
        )

    def expand_passages(
        self,
        candidates: tuple[
            IndexedAyahCandidate,
            ...,
        ],
        *,
        before: int = 2,
        after: int = 2,
    ) -> tuple[tuple[int, int], ...]:
        """
        Expand shortlisted ayahs into nearby references.

        This allows the detailed matcher to evaluate consecutive passages
        surrounding each strong starting candidate.
        """

        references: set[tuple[int, int]] = set()

        for candidate in candidates:
            surah = self.corpus.get_surah(
                candidate.surah_number
            )

            if not surah:
                continue

            first_ayah = max(
                1,
                candidate.ayah_number - before,
            )

            last_ayah = min(
                len(surah),
                candidate.ayah_number + after,
            )

            for ayah_number in range(
                first_ayah,
                last_ayah + 1,
            ):
                references.add(
                    (
                        candidate.surah_number,
                        ayah_number,
                    )
                )

        return tuple(
            sorted(references)
        )
