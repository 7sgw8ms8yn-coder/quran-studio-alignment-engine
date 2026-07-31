from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.services.quran_corpus_service import QuranCorpus


_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

_NON_ARABIC = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")


def normalise_for_matching(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }

    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    text = _NON_ARABIC.sub(" ", text)
    return " ".join(text.split())


def phonetic_normalise(text: str) -> str:
    text = normalise_for_matching(text)

    replacements = {
        "ث": "س",
        "ص": "س",
        "ذ": "ز",
        "ظ": "ز",
        "ض": "د",
        "ط": "ت",
        "ق": "ك",
        "غ": "خ",
    }

    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    return text


def character_ngrams(
    text: str,
    size: int = 3,
) -> set[str]:
    compact = text.replace(" ", "")

    if not compact:
        return set()

    if len(compact) <= size:
        return {compact}

    return {
        compact[index:index + size]
        for index in range(
            len(compact) - size + 1
        )
    }


def dice_similarity(
    first: set[str],
    second: set[str],
) -> float:
    if not first or not second:
        return 0.0

    return (
        2.0 * len(first & second)
        / (len(first) + len(second))
    )


def fuzzy_word_coverage(
    transcript_words: list[str],
    verified_words: list[str],
    threshold: float = 0.66,
) -> float:
    if not transcript_words or not verified_words:
        return 0.0

    matched = 0

    for transcript_word in transcript_words:
        best_similarity = max(
            SequenceMatcher(
                None,
                transcript_word,
                verified_word,
                autojunk=False,
            ).ratio()
            for verified_word in verified_words
        )

        if best_similarity >= threshold:
            matched += 1

    return matched / len(transcript_words)


@dataclass(frozen=True, slots=True)
class QuranCandidateMatch:
    surah_number: int
    start_ayah: int
    end_ayah: int
    score: float
    sequence_similarity: float
    word_overlap: float
    transcript_coverage: float
    fuzzy_coverage: float
    ngram_similarity: float
    phonetic_similarity: float
    length_compatibility: float
    matched_text: str
    normalised_transcript: str

    @property
    def ayah_count(self) -> int:
        return self.end_ayah - self.start_ayah + 1


class QuranCandidateMatcher:
    def __init__(
        self,
        corpus: QuranCorpus,
        max_sequence_length: int = 4,
    ) -> None:
        if max_sequence_length < 1:
            raise ValueError(
                "max_sequence_length must be at least 1."
            )

        self.corpus = corpus
        self.max_sequence_length = max_sequence_length

    def find_candidates(
        self,
        transcript: str,
        limit: int = 5,
    ) -> tuple[QuranCandidateMatch, ...]:
        normalised_transcript = normalise_for_matching(
            transcript
        )

        if len(normalised_transcript) < 3:
            return ()

        transcript_words = normalised_transcript.split()

        if not transcript_words:
            return ()

        transcript_word_set = set(transcript_words)
        transcript_ngrams = character_ngrams(
            normalised_transcript
        )

        phonetic_transcript = phonetic_normalise(
            normalised_transcript
        )

        safe_limit = max(1, min(limit, 25))
        matches: list[QuranCandidateMatch] = []

        by_surah: dict[int, list[object]] = {}

        for ayah in self.corpus.ayahs:
            by_surah.setdefault(
                ayah.surah_number,
                [],
            ).append(ayah)

        for surah_number, ayahs in by_surah.items():
            ayahs.sort(
                key=lambda item: item.ayah_number
            )

            for start_index in range(len(ayahs)):
                maximum = min(
                    self.max_sequence_length,
                    len(ayahs) - start_index,
                )

                for sequence_length in range(
                    1,
                    maximum + 1,
                ):
                    sequence = ayahs[
                        start_index:
                        start_index + sequence_length
                    ]

                    verified_text = " ".join(
                        ayah.text for ayah in sequence
                    )

                    normalised_verified = (
                        normalise_for_matching(
                            verified_text
                        )
                    )

                    verified_words = (
                        normalised_verified.split()
                    )

                    if not verified_words:
                        continue

                    verified_word_set = set(
                        verified_words
                    )

                    shared_words = (
                        transcript_word_set
                        & verified_word_set
                    )

                    union_words = (
                        transcript_word_set
                        | verified_word_set
                    )

                    sequence_similarity = (
                        SequenceMatcher(
                            None,
                            normalised_transcript,
                            normalised_verified,
                            autojunk=False,
                        ).ratio()
                    )

                    word_overlap = (
                        len(shared_words)
                        / len(union_words)
                        if union_words
                        else 0.0
                    )

                    transcript_coverage = (
                        len(shared_words)
                        / len(transcript_word_set)
                        if transcript_word_set
                        else 0.0
                    )

                    fuzzy_coverage = fuzzy_word_coverage(
                        transcript_words,
                        verified_words,
                    )

                    ngram_similarity = dice_similarity(
                        transcript_ngrams,
                        character_ngrams(
                            normalised_verified
                        ),
                    )

                    phonetic_similarity = (
                        SequenceMatcher(
                            None,
                            phonetic_transcript,
                            phonetic_normalise(
                                normalised_verified
                            ),
                            autojunk=False,
                        ).ratio()
                    )

                    shorter_length = min(
                        len(transcript_words),
                        len(verified_words),
                    )

                    longer_length = max(
                        len(transcript_words),
                        len(verified_words),
                    )

                    length_compatibility = (
                        shorter_length / longer_length
                        if longer_length
                        else 0.0
                    )

                    broad_range_penalty = (
                        max(0, sequence_length - 1)
                        * 0.012
                    )

                    combined_score = (
                        sequence_similarity * 0.18
                        + word_overlap * 0.08
                        + transcript_coverage * 0.12
                        + fuzzy_coverage * 0.22
                        + ngram_similarity * 0.18
                        + phonetic_similarity * 0.16
                        + length_compatibility * 0.06
                        - broad_range_penalty
                    )

                    combined_score = max(
                        0.0,
                        min(1.0, combined_score),
                    )

                    matches.append(
                        QuranCandidateMatch(
                            surah_number=surah_number,
                            start_ayah=(
                                sequence[0].ayah_number
                            ),
                            end_ayah=(
                                sequence[-1].ayah_number
                            ),
                            score=round(
                                combined_score,
                                6,
                            ),
                            sequence_similarity=round(
                                sequence_similarity,
                                6,
                            ),
                            word_overlap=round(
                                word_overlap,
                                6,
                            ),
                            transcript_coverage=round(
                                transcript_coverage,
                                6,
                            ),
                            fuzzy_coverage=round(
                                fuzzy_coverage,
                                6,
                            ),
                            ngram_similarity=round(
                                ngram_similarity,
                                6,
                            ),
                            phonetic_similarity=round(
                                phonetic_similarity,
                                6,
                            ),
                            length_compatibility=round(
                                length_compatibility,
                                6,
                            ),
                            matched_text=verified_text,
                            normalised_transcript=(
                                normalised_transcript
                            ),
                        )
                    )

        matches.sort(
            key=lambda match: (
                match.score,
                match.fuzzy_coverage,
                match.ngram_similarity,
                match.length_compatibility,
                -match.ayah_count,
            ),
            reverse=True,
        )

        return tuple(matches[:safe_limit])

    def best_match(
        self,
        transcript: str,
        minimum_score: float = 0.58,
        minimum_fuzzy_coverage: float = 0.45,
        minimum_ngram_similarity: float = 0.28,
        minimum_margin: float = 0.025,
    ) -> QuranCandidateMatch | None:
        candidates = self.find_candidates(
            transcript,
            limit=2,
        )

        if not candidates:
            return None

        best = candidates[0]

        if best.score < minimum_score:
            return None

        if (
            best.fuzzy_coverage
            < minimum_fuzzy_coverage
        ):
            return None

        if (
            best.ngram_similarity
            < minimum_ngram_similarity
        ):
            return None

        if len(candidates) > 1:
            runner_up = candidates[1]
            score_margin = (
                best.score - runner_up.score
            )

            same_passage = (
                best.surah_number
                == runner_up.surah_number
                and best.start_ayah
                <= runner_up.end_ayah
                and runner_up.start_ayah
                <= best.end_ayah
            )

            if (
                score_margin < minimum_margin
                and not same_passage
            ):
                return None

        return best
