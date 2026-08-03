from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests.auth import HTTPBasicAuth


class QuranFoundationError(RuntimeError):
    """Raised when Quran Foundation cannot complete a request safely."""


@dataclass(frozen=True, slots=True)
class QuranFoundationVerse:
    verse_key: str
    chapter_id: int
    verse_number: int
    text_uthmani: str | None
    text_imlaei: str | None
    words: tuple[dict[str, Any], ...]
    raw: dict[str, Any]


class QuranFoundationService:
    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        environment: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.client_id = (
            client_id or os.getenv("QF_CLIENT_ID", "")
        ).strip()

        self.client_secret = (
            client_secret or os.getenv("QF_CLIENT_SECRET", "")
        ).strip()

        self.environment = (
            environment or os.getenv("QF_ENV", "production")
        ).strip().lower()

        if not self.client_id:
            raise QuranFoundationError(
                "QF_CLIENT_ID is not configured."
            )

        if not self.client_secret:
            raise QuranFoundationError(
                "QF_CLIENT_SECRET is not configured."
            )

        if self.environment not in {"production", "prelive"}:
            raise QuranFoundationError(
                "QF_ENV must be 'production' or 'prelive'."
            )

        self.timeout_seconds = timeout_seconds

        if self.environment == "production":
            self.auth_base_url = (
                "https://oauth2.quran.foundation"
            )
            self.api_base_url = (
                "https://apis.quran.foundation"
            )
        else:
            self.auth_base_url = (
                "https://prelive-oauth2.quran.foundation"
            )
            self.api_base_url = (
                "https://apis-prelive.quran.foundation"
            )

        self._session = requests.Session()
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> QuranFoundationService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request_new_token(self) -> str:
        response = self._session.post(
            f"{self.auth_base_url}/oauth2/token",
            auth=HTTPBasicAuth(
                self.client_id,
                self.client_secret,
            ),
            headers={
                "Content-Type":
                    "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "content",
            },
            timeout=self.timeout_seconds,
        )

        if not response.ok:
            raise QuranFoundationError(
                "Quran Foundation authentication failed "
                f"with status {response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            payload = response.json()
            access_token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 3600))
        except (ValueError, KeyError, TypeError) as exc:
            raise QuranFoundationError(
                "Quran Foundation returned an invalid token response."
            ) from exc

        if not isinstance(access_token, str) or not access_token:
            raise QuranFoundationError(
                "Quran Foundation returned an empty access token."
            )

        # Renew one minute early to avoid expiry during an API request.
        safe_lifetime = max(60, expires_in - 60)

        self._access_token = access_token
        self._token_expires_at = (
            time.monotonic() + safe_lifetime
        )

        return access_token

    def _get_access_token(
        self,
        *,
        force_refresh: bool = False,
    ) -> str:
        with self._token_lock:
            token_is_valid = (
                self._access_token is not None
                and time.monotonic() < self._token_expires_at
            )

            if token_is_valid and not force_refresh:
                return self._access_token

            return self._request_new_token()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(2):
            token = self._get_access_token(
                force_refresh=attempt == 1
            )

            response = self._session.request(
                method,
                f"{self.api_base_url}{path}",
                headers={
                    "x-auth-token": token,
                    "x-client-id": self.client_id,
                    "Accept": "application/json",
                },
                params=params,
                timeout=self.timeout_seconds,
            )

            if response.status_code == 401 and attempt == 0:
                continue

            if not response.ok:
                raise QuranFoundationError(
                    "Quran Foundation request failed "
                    f"with status {response.status_code}: "
                    f"{response.text[:500]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise QuranFoundationError(
                    "Quran Foundation returned invalid JSON."
                ) from exc

            if not isinstance(payload, dict):
                raise QuranFoundationError(
                    "Quran Foundation returned an unexpected response."
                )

            return payload

        raise QuranFoundationError(
            "Quran Foundation authentication retry failed."
        )

    def get_chapters(self) -> tuple[dict[str, Any], ...]:
        payload = self._request(
            "GET",
            "/content/api/v4/chapters",
        )

        chapters = payload.get("chapters", [])

        if not isinstance(chapters, list):
            raise QuranFoundationError(
                "Chapters response has an invalid format."
            )

        return tuple(
            chapter
            for chapter in chapters
            if isinstance(chapter, dict)
        )

    def get_verse(
        self,
        surah_number: int,
        ayah_number: int,
        *,
        include_words: bool = True,
    ) -> QuranFoundationVerse:
        if not 1 <= surah_number <= 114:
            raise ValueError(
                "surah_number must be between 1 and 114."
            )

        if ayah_number < 1:
            raise ValueError(
                "ayah_number must be at least 1."
            )

        verse_key = f"{surah_number}:{ayah_number}"

        payload = self._request(
            "GET",
            f"/content/api/v4/verses/by_key/{verse_key}",
            params={
                "language": "en",
                "words": str(include_words).lower(),
                "fields": (
                    "text_uthmani,text_imlaei,"
                    "chapter_id,verse_number,verse_key"
                ),
            },
        )

        verse = payload.get("verse")

        if not isinstance(verse, dict):
            raise QuranFoundationError(
                f"Verse {verse_key} was not returned."
            )

        words = verse.get("words", [])

        if not isinstance(words, list):
            words = []

        return QuranFoundationVerse(
            verse_key=str(
                verse.get("verse_key", verse_key)
            ),
            chapter_id=int(
                verse.get("chapter_id", surah_number)
            ),
            verse_number=int(
                verse.get("verse_number", ayah_number)
            ),
            text_uthmani=verse.get("text_uthmani"),
            text_imlaei=verse.get("text_imlaei"),
            words=tuple(
                word
                for word in words
                if isinstance(word, dict)
            ),
            raw=verse,
        )
