import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path


class MediaPreparationError(RuntimeError):
    """Raised when media cannot be prepared for analysis."""


@dataclass(frozen=True)
class PreparedAudio:
    source_path: Path
    audio_path: Path
    sample_rate: int
    channels: int
    codec: str


def ffmpeg_is_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_is_available() -> bool:
    return shutil.which("ffprobe") is not None


async def _run_command(
    *command: str,
) -> tuple[str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_bytes, stderr_bytes = await process.communicate()

    stdout = stdout_bytes.decode(
        "utf-8",
        errors="replace",
    )

    stderr = stderr_bytes.decode(
        "utf-8",
        errors="replace",
    )

    if process.returncode != 0:
        raise MediaPreparationError(
            stderr.strip()
            or "The media-processing command failed."
        )

    return stdout, stderr


async def prepare_audio_for_alignment(
    source_path: Path,
    output_path: Path,
) -> PreparedAudio:
    if not source_path.exists():
        raise MediaPreparationError(
            "The uploaded source media does not exist."
        )

    if not ffmpeg_is_available():
        raise MediaPreparationError(
            "FFmpeg is not installed."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.unlink(missing_ok=True)

    await _run_command(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    )

    if not output_path.exists():
        raise MediaPreparationError(
            "FFmpeg did not create the prepared audio file."
        )

    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)

        raise MediaPreparationError(
            "The prepared audio file is empty."
        )

    return PreparedAudio(
        source_path=source_path,
        audio_path=output_path,
        sample_rate=16000,
        channels=1,
        codec="pcm_s16le",
    )


async def read_duration_seconds(
    media_path: Path,
) -> float:
    if not media_path.exists():
        raise MediaPreparationError(
            "The media file does not exist."
        )

    if not ffprobe_is_available():
        raise MediaPreparationError(
            "FFprobe is not installed."
        )

    stdout, _ = await _run_command(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    )

    try:
        duration = float(stdout.strip())
    except ValueError as error:
        raise MediaPreparationError(
            "The media duration could not be read."
        ) from error

    if duration <= 0:
        raise MediaPreparationError(
            "The media duration is invalid."
        )

    return duration