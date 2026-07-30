from pathlib import Path

import pytest

from app.services.media_service import (
    MediaPreparationError,
    ffmpeg_is_available,
    ffprobe_is_available,
    prepare_audio_for_alignment,
)


def test_ffmpeg_is_available() -> None:
    assert isinstance(
        ffmpeg_is_available(),
        bool,
    )


def test_ffprobe_is_available() -> None:
    assert isinstance(
        ffprobe_is_available(),
        bool,
    )


@pytest.mark.anyio
async def test_missing_source_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing.mp3"
    destination = tmp_path / "prepared.wav"

    with pytest.raises(
        MediaPreparationError,
        match="does not exist",
    ):
        await prepare_audio_for_alignment(
            source,
            destination,
        )