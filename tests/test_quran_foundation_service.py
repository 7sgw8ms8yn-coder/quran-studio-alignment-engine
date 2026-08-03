from unittest.mock import Mock

import pytest

from app.services.quran_foundation_service import (
    QuranFoundationError,
    QuranFoundationService,
)


def make_service() -> QuranFoundationService:
    return QuranFoundationService(
        client_id="test-client",
        client_secret="test-secret",
        environment="production",
    )


def test_rejects_invalid_environment() -> None:
    with pytest.raises(QuranFoundationError):
        QuranFoundationService(
            client_id="client",
            client_secret="secret",
            environment="invalid",
        )


def test_get_verse_parses_verified_text() -> None:
    service = make_service()

    service._request = Mock(
        return_value={
            "verse": {
                "verse_key": "2:102",
                "chapter_id": 2,
                "verse_number": 102,
                "text_uthmani": "وَٱتَّبَعُوا۟",
                "text_imlaei": "واتبعوا",
                "words": [
                    {
                        "position": 1,
                        "text_uthmani": "وَٱتَّبَعُوا۟",
                    }
                ],
            }
        }
    )

    verse = service.get_verse(2, 102)

    assert verse.verse_key == "2:102"
    assert verse.chapter_id == 2
    assert verse.verse_number == 102
    assert verse.text_uthmani == "وَٱتَّبَعُوا۟"
    assert len(verse.words) == 1


def test_get_verse_rejects_invalid_surah() -> None:
    service = make_service()

    with pytest.raises(ValueError):
        service.get_verse(115, 1)


def test_get_verse_rejects_invalid_ayah() -> None:
    service = make_service()

    with pytest.raises(ValueError):
        service.get_verse(1, 0)
