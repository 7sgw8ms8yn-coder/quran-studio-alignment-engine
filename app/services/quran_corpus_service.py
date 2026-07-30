from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXPECTED_AYAH_COUNT = 6236

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "quran"
    / "quran-simple-clean.txt"
)

# Standard number of Ayahs in each Surah, ordered from 1 to 114.
SURAH_AYAH_COUNTS: tuple[int, ...] = (
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
    123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
    34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
    60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
    15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
    5, 4, 5, 6,
)

ARABIC_DIACRITICS = re.compile(
    "["
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06ED"
    "]"
)

NON_ARABIC_OR_SPACE = re.compile(r"[^\u0621-\u063A\u0641-\u064A ]+")
MULTIPLE_SPACES = re.compile(r"\s+")


class QuranCorpusError(RuntimeError):
    """Raised when the Quran corpus is missing or invalid."""


@dataclass(frozen=True, slots=True)
class QuranAyah:
    surah_number: int
    ayah_number: int
    text: str
    normalized_text: str

    @property
    def key(self) -> str:
        return f"{self.surah_number}:{self.ayah_number}"


@dataclass(frozen=True, slots=True)
class QuranCorpus:
    ayahs: tuple[QuranAyah, ...]

    def get_ayah(
        self,
        surah_number: int,
        ayah_number: int,
    ) -> QuranAyah:
        for ayah in self.ayahs:
            if (
                ayah.surah_number == surah_number
                and ayah.ayah_number == ayah_number
            ):
                return ayah

        raise KeyError(
            f"Ayah {surah_number}:{ayah_number} was not found."
        )

    def get_surah(
        self,
        surah_number: int,
    ) -> tuple[QuranAyah, ...]:
        return tuple(
            ayah
            for ayah in self.ayahs
            if ayah.surah_number == surah_number
        )

    def search_exact(
        self,
        arabic_text: str,
    ) -> tuple[QuranAyah, ...]:
        query = normalize_arabic_for_search(arabic_text)

        if not query:
            return ()

        return tuple(
            ayah
            for ayah in self.ayahs
            if ayah.normalized_text == query
        )

    def search_contains(
        self,
        arabic_text: str,
    ) -> tuple[QuranAyah, ...]:
        query = normalize_arabic_for_search(arabic_text)

        if not query:
            return ()

        return tuple(
            ayah
            for ayah in self.ayahs
            if query in ayah.normalized_text
        )


def normalize_arabic_for_search(text: str) -> str:
    """
    Create a search-only form of Arabic text.

    This function never changes the stored Quran corpus text.
    """

    normalized = unicodedata.normalize("NFKC", text)

    normalized = ARABIC_DIACRITICS.sub("", normalized)

    character_map = str.maketrans(
        {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ؤ": "و",
            "ئ": "ي",
            "ـ": "",
        }
    )

    normalized = normalized.translate(character_map)
    normalized = NON_ARABIC_OR_SPACE.sub(" ", normalized)
    normalized = MULTIPLE_SPACES.sub(" ", normalized)

    return normalized.strip()


def _iter_content_lines(
    corpus_path: Path,
) -> Iterable[str]:
    with corpus_path.open(
        "r",
        encoding="utf-8-sig",
    ) as corpus_file:
        for raw_line in corpus_file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            yield line


def _generate_standard_references() -> list[tuple[int, int]]:
    references: list[tuple[int, int]] = []

    for surah_number, ayah_count in enumerate(
        SURAH_AYAH_COUNTS,
        start=1,
    ):
        for ayah_number in range(1, ayah_count + 1):
            references.append(
                (surah_number, ayah_number)
            )

    return references


def _parse_numbered_line(
    line: str,
) -> tuple[int, int, str] | None:
    parts = line.split("|", maxsplit=2)

    if len(parts) != 3:
        return None

    surah_text, ayah_text, quran_text = parts

    try:
        surah_number = int(surah_text)
        ayah_number = int(ayah_text)
    except ValueError:
        return None

    return (
        surah_number,
        ayah_number,
        quran_text.strip(),
    )


def load_quran_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
) -> QuranCorpus:
    if not corpus_path.exists():
        raise QuranCorpusError(
            f"Quran corpus file is missing: {corpus_path}"
        )

    lines = list(_iter_content_lines(corpus_path))

    if len(lines) != EXPECTED_AYAH_COUNT:
        raise QuranCorpusError(
            "The Quran corpus must contain exactly "
            f"{EXPECTED_AYAH_COUNT} Ayahs, but "
            f"{len(lines)} were found."
        )

    first_parsed = _parse_numbered_line(lines[0])
    references = _generate_standard_references()

    ayahs: list[QuranAyah] = []

    for index, line in enumerate(lines):
        parsed = _parse_numbered_line(line)

        if first_parsed is not None:
            if parsed is None:
                raise QuranCorpusError(
                    f"Invalid numbered corpus line {index + 1}."
                )

            surah_number, ayah_number, text = parsed
        else:
            surah_number, ayah_number = references[index]
            text = line

        if not 1 <= surah_number <= 114:
            raise QuranCorpusError(
                f"Invalid Surah number on line {index + 1}."
            )

        maximum_ayah = SURAH_AYAH_COUNTS[surah_number - 1]

        if not 1 <= ayah_number <= maximum_ayah:
            raise QuranCorpusError(
                f"Invalid Ayah number on line {index + 1}."
            )

        expected_surah, expected_ayah = references[index]

        if (
            surah_number != expected_surah
            or ayah_number != expected_ayah
        ):
            raise QuranCorpusError(
                "Unexpected Quran ordering on line "
                f"{index + 1}: expected "
                f"{expected_surah}:{expected_ayah}, found "
                f"{surah_number}:{ayah_number}."
            )

        if not text:
            raise QuranCorpusError(
                f"Empty Quran text on line {index + 1}."
            )

        normalized_text = normalize_arabic_for_search(text)

        if not normalized_text:
            raise QuranCorpusError(
                "The Quran text could not be normalised "
                f"on line {index + 1}."
            )

        ayahs.append(
            QuranAyah(
                surah_number=surah_number,
                ayah_number=ayah_number,
                text=text,
                normalized_text=normalized_text,
            )
        )

    return QuranCorpus(
        ayahs=tuple(ayahs),
    )