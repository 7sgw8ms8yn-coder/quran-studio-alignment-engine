import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.job_store import AlignmentJob, job_store
from app.models import (
    CancelJobResponse,
    HealthResponse,
    JobAcceptedResponse,
    JobStatusResponse,
)
from app.security import verify_api_key


AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
}

ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.temporary_directory).mkdir(
        parents=True,
        exist_ok=True,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.engine_version,
    description=(
        "Quran-constrained audio alignment service "
        "for Quran Studio AI."
    ),
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


def serialise_job(
    job: AlignmentJob,
) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        client_job_id=job.client_job_id,
        project_id=job.project_id,
        owner_id=job.owner_id,
        status=job.status,  # type: ignore[arg-type]
        progress=job.progress,
        current_stage=job.current_stage,
        message=job.message,
        warnings=job.warnings,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def validate_settings(
    settings_json: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(settings_json)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The settings field is not valid JSON.",
        ) from error

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The settings field must contain a JSON object.",
        )

    required_fields = {
        "projectId",
        "clientJobId",
        "ownerId",
        "detectionMode",
        "timingMode",
        "qualityMode",
        "mediaType",
    }

    missing_fields = sorted(
        field
        for field in required_fields
        if not payload.get(field)
    )

    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Required detection settings are missing.",
                "missing_fields": missing_fields,
            },
        )

    return payload


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.engine_version,
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    ready = (
        settings.model_loaded
        and settings.corpus_loaded
    )

    return HealthResponse(
        status="healthy" if ready else "not_ready",
        engine_version=settings.engine_version,
        corpus_loaded=settings.corpus_loaded,
        model_loaded=settings.model_loaded,
        message=(
            "The Quran Alignment Engine is ready."
            if ready
            else (
                "The API is online. The Quran corpus and "
                "speech model will be connected next."
            )
        ),
    )


@app.post(
    "/jobs",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
)
async def create_job(
    file: Annotated[UploadFile, File()],
    settings_json: Annotated[
        str,
        Form(alias="settings"),
    ],
) -> JobAcceptedResponse:
    filename = Path(
        file.filename or "upload"
    ).name

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio or video format.",
        )

    detection_settings = validate_settings(settings_json)

    maximum_bytes = (
        settings.maximum_upload_mb * 1024 * 1024
    )

    received_bytes = 0

    while True:
        chunk = await file.read(1024 * 1024)

        if not chunk:
            break

        received_bytes += len(chunk)

        if received_bytes > maximum_bytes:
            await file.close()

            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "The initial alignment engine accepts files "
                    f"up to {settings.maximum_upload_mb} MB."
                ),
            )

    await file.close()

    if received_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded media file is empty.",
        )

    job = job_store.create(
        client_job_id=str(
            detection_settings["clientJobId"]
        ),
        project_id=str(
            detection_settings["projectId"]
        ),
        owner_id=str(
            detection_settings["ownerId"]
        ),
        original_filename=filename,
    )

    return JobAcceptedResponse(
        job_id=job.id,
        client_job_id=job.client_job_id,
        project_id=job.project_id,
        status="engine_not_ready",
        created_at=job.created_at,
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
async def get_job(
    job_id: str,
) -> JobStatusResponse:
    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alignment job not found.",
        )

    return serialise_job(job)


@app.post(
    "/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    dependencies=[Depends(verify_api_key)],
)
async def cancel_job(
    job_id: str,
) -> CancelJobResponse:
    job = job_store.cancel(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alignment job not found.",
        )

    return CancelJobResponse(
        job_id=job.id,
        status="cancelled",
        message=job.message,
    )