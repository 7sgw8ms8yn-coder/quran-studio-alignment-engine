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


def select_partial_ayah_scoring_window(
    transcript_words: list[str],
    verified_words: list[str],
) -> list[str]:
    if not transcript_words or not verified_words:
        return verified_words

    # Very short Quran phrases can legitimately occur inside multiple Ayahs.
    # Do not use partial-Ayah window scoring for such short transcripts,
    # because doing so can make several different Ayahs appear to be
    # identical perfect matches.
    if len(transcript_words) < 8:
        return verified_words

    if len(verified_words) <= len(transcript_words) + 8:
        return verified_words

    matcher = SequenceMatcher(
        None,
        transcript_words,
        verified_words,
        autojunk=False,
    )

    block = matcher.find_longest_match(
        0,
        len(transcript_words),
        0,
        len(verified_words),
    )

    # Do not switch to partial-Ayah scoring unless there is a
    # substantial contiguous anchor.
    minimum_anchor = max(
        4,
        int(round(len(transcript_words) * 0.35)),
    )

    if block.size < minimum_anchor:
        return verified_words

    estimated_start = max(
        0,
        block.b - block.a,
    )

    # Score the candidate against the Quran span corresponding to the
    # amount of speech actually present in the uploaded clip.
    #
    # The longest-match offset estimates where a partial recitation starts
    # inside a long Ayah. We deliberately avoid adding arbitrary Quran words
    # before or after that span because unrecited words must not lower Quran
    # identification confidence.
    start = min(
        estimated_start,
        max(0, len(verified_words) - 1),
    )

    end = min(
        len(verified_words),
        start + len(transcript_words),
    )

    # Keep a usable window if the estimated start lands very near the end.
    if end - start < min(len(transcript_words), len(verified_words)):
        start = max(
            0,
            end - min(
                len(transcript_words),
                len(verified_words),
            ),
        )

    return verified_words[start:end]


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

        # Local import avoids a circular import while the index reuses
        # this module's Quran normalisation functions.
        from app.services.quran_candidate_index import QuranCandidateIndex

        self.candidate_index = QuranCandidateIndex(corpus)

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

        # Stage 1: retrieve likely ayahs in milliseconds.
        # Keep the retrieval pool focused. Detailed fuzzy scoring is much
        # more expensive than the index search, so only the strongest
        # first-stage candidates should reach it.
        shortlist = self.candidate_index.shortlist(
            transcript,
            limit=max(15, safe_limit * 3),
        )

        if not shortlist:
            return ()

        # Include surrounding ayahs so multi-ayah recitations remain possible.
        shortlisted_references = set(
            self.candidate_index.expand_passages(
                shortlist,
                before=1,
                after=min(
                    6,
                    max(2, self.max_sequence_length),
                ),
            )
        )

        by_surah: dict[int, list[object]] = {}

        for ayah in self.corpus.ayahs:
            reference = (
                ayah.surah_number,
                ayah.ayah_number,
            )

            if reference not in shortlisted_references:
                continue

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

                    # Do not combine ayahs separated by a shortlist gap.
                    if any(
                        current.ayah_number + 1
                        != following.ayah_number
                        for current, following in zip(
                            sequence,
                            sequence[1:],
                        )
                    ):
                        continue

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

                    scoring_words = select_partial_ayah_scoring_window(
                        transcript_words,
                        verified_words,
                    )

                    scoring_text = " ".join(scoring_words)

                    verified_word_set = set(
                        scoring_words
                    )

                    shared_words = (
                        transcript_word_set
                        & verified_word_set
                    )

                    union_words = (
                        transcript_word_set
                        | verified_word_set
                    )

                    transcript_tokens = normalised_transcript.split()
                    verified_tokens = scoring_words

                    sequence_similarity = (
                        SequenceMatcher(
                            None,
                            transcript_tokens,
                            verified_tokens,
                            autojunk=False,
                        ).ratio()
                        if transcript_tokens and verified_tokens
                        else 0.0
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
                        scoring_words,
                    )

                    ngram_similarity = dice_similarity(
                        transcript_ngrams,
                        character_ngrams(
                            scoring_text
                        ),
                    )

                    phonetic_verified = phonetic_normalise(scoring_text)

                    phonetic_transcript_tokens = set(phonetic_transcript.split())
                    phonetic_verified_tokens = set(phonetic_verified.split())

                    phonetic_total = (
                        len(phonetic_transcript_tokens)
                        + len(phonetic_verified_tokens)
                    )

                    phonetic_similarity = (
                        2.0
                        * len(phonetic_transcript_tokens & phonetic_verified_tokens)
                        / phonetic_total
                        if phonetic_total
                        else 0.0
                    )

                    shorter_length = min(
                        len(transcript_words),
                        len(scoring_words),
                    )

                    longer_length = max(
                        len(transcript_words),
                        len(scoring_words),
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
