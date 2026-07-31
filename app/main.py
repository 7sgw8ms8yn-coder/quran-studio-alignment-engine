import asyncio
import shutil
import tempfile
from uuid import uuid4
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    BackgroundTasks,
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
from app.services.media_service import (
    MediaPreparationError,
    prepare_audio_for_alignment,
)
from app.services.quran_alignment_service import (
    QuranAlignmentError,
    QuranAlignmentService,
)
from app.services.speech_recognition_service import ArabicSpeechRecognizer
from app.services.quran_corpus_service import (
    QuranCorpus,
    QuranCorpusError,
    load_quran_corpus,
)
from app.services.speech_recognition_service import speech_recognizer

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


quran_speech_recognizer = ArabicSpeechRecognizer(
    model_name="OdyAsh/faster-whisper-base-ar-quran",
    device="cpu",
    compute_type="int8",
)

class EngineState:
    def __init__(self) -> None:
        self.quran_corpus: QuranCorpus | None = None
        self.corpus_error: str | None = None

    @property
    def corpus_loaded(self) -> bool:
        return self.quran_corpus is not None


engine_state = EngineState()
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.temporary_directory).mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        engine_state.quran_corpus = load_quran_corpus()
        engine_state.corpus_error = None
    except QuranCorpusError as error:
        engine_state.quran_corpus = None
        engine_state.corpus_error = str(error)

    app.state.quran_corpus = engine_state.quran_corpus

    yield

    engine_state.quran_corpus = None
    app.state.quran_corpus = None


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
    corpus_loaded = engine_state.corpus_loaded
    model_loaded = settings.model_loaded

    fully_ready = corpus_loaded and model_loaded

    if fully_ready:
        message = "The Quran Alignment Engine is ready."
    elif not corpus_loaded:
        message = (
            "The API is online, but the verified Quran corpus "
            "could not be loaded."
        )
    else:
        message = (
            "The API and verified Quran corpus are ready. "
            "The speech-recognition model will be connected next."
        )

    return HealthResponse(
        status="healthy" if fully_ready else "not_ready",
        engine_version=settings.engine_version,
        corpus_loaded=corpus_loaded,
        model_loaded=model_loaded,
        message=message,
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
@app.get(
    "/quran/{surah_number}/{ayah_number}",
    dependencies=[Depends(verify_api_key)],
)
async def get_quran_ayah(
    surah_number: int,
    ayah_number: int,
) -> dict[str, object]:
    corpus = engine_state.quran_corpus

    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The Quran corpus is not available.",
        )

    try:
        ayah = corpus.get_ayah(
            surah_number,
            ayah_number,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested Ayah was not found.",
        ) from error

    return {
        "surah_number": ayah.surah_number,
        "ayah_number": ayah.ayah_number,
        "text": ayah.text,
    }


@app.get(
    "/quran/search",
    dependencies=[Depends(verify_api_key)],
)
async def search_quran(
    query: str,
    limit: int = 10,
) -> dict[str, object]:
    corpus = engine_state.quran_corpus

    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The Quran corpus is not available.",
        )

    cleaned_query = query.strip()

    if len(cleaned_query) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enter at least three Arabic characters.",
        )

    safe_limit = max(1, min(limit, 25))
    matches = corpus.search_contains(cleaned_query)[:safe_limit]

    return {
        "query": cleaned_query,
        "count": len(matches),
        "results": [
            {
                "surah_number": ayah.surah_number,
                "ayah_number": ayah.ayah_number,
                "text": ayah.text,
            }
            for ayah in matches
        ],
    }


@app.get(
    "/speech/model-status",
    dependencies=[Depends(verify_api_key)],
)
async def speech_model_status() -> dict[str, object]:
    return {
        "model_name": speech_recognizer.model_name,
        "device": speech_recognizer.device,
        "compute_type": speech_recognizer.compute_type,
        "loaded": speech_recognizer.is_loaded,
        "load_error": speech_recognizer.load_error,
    }


async def _execute_quran_alignment(
    file: UploadFile = File(...),
) -> dict[str, object]:
    filename = file.filename or "uploaded-media"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported media format. "
                "Upload a supported audio or video file."
            ),
        )

    corpus = getattr(
        app.state,
        "quran_corpus",
        None,
    )

    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The verified Quran corpus is unavailable.",
        )

    work_directory = (
        Path(settings.temporary_directory)
        / f"quran-alignment-{uuid4().hex}"
    )

    work_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    source_path = (
        work_directory
        / f"source{extension}"
    )

    prepared_path = (
        work_directory
        / "prepared.wav"
    )

    try:
        with source_path.open("wb") as destination:
            shutil.copyfileobj(
                file.file,
                destination,
            )

        if source_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The uploaded media file is empty.",
            )

        await prepare_audio_for_alignment(
            source_path,
            prepared_path,
        )

        alignment_service = QuranAlignmentService(
            corpus=corpus,
            recognizer=quran_speech_recognizer,
            max_sequence_length=6,
            minimum_match_score=0.58,
        )

        result = await asyncio.to_thread(
            alignment_service.align,
            prepared_path,
        )

    except MediaPreparationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except QuranAlignmentError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    finally:
        await file.close()

        shutil.rmtree(
            work_directory,
            ignore_errors=True,
        )

    return {
        "surah_number": result.surah_number,
        "start_ayah": result.start_ayah,
        "end_ayah": result.end_ayah,
        "match_score": result.match_score,
        "duration": result.duration,
        "rough_transcript": result.transcript,
        "verified_text": result.verified_text,
        "ayahs": [
            {
                "surah_number": ayah.surah_number,
                "ayah_number": ayah.ayah_number,
                "start": ayah.start,
                "end": ayah.end,
                "text": ayah.text,
            }
            for ayah in result.ayahs
        ],
    }

# --- ASYNC QURAN ALIGNMENT JOB FLOW ---


async def _process_quran_alignment_job(
    job_id: str,
    source_path_text: str,
    original_filename: str,
    staging_directory_text: str,
) -> None:
    """Run Quran alignment after the upload request has returned."""

    source_path = Path(source_path_text)
    staging_directory = Path(staging_directory_text)
    upload: UploadFile | None = None

    try:
        job_store.start(job_id)

        job_store.update(
            job_id,
            progress=10,
            current_stage="preparing_audio",
            message="Preparing the uploaded recitation for Quran alignment.",
        )

        source_handle = source_path.open("rb")
        upload = UploadFile(
            file=source_handle,
            filename=original_filename,
        )

        job_store.update(
            job_id,
            progress=25,
            current_stage="transcribing",
            message="Transcribing and analysing the Quran recitation.",
        )

        result = await _execute_quran_alignment(upload)

        job_store.update(
            job_id,
            progress=90,
            current_stage="verifying_quran_text",
            message="Verifying the transcript against the Quran corpus.",
        )

        job_store.complete(job_id, result)

    except Exception as error:
        job_store.fail(
            job_id,
            f"{type(error).__name__}: {error}",
        )

    finally:
        if upload is not None:
            try:
                await upload.close()
            except Exception:
                pass

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )


@app.post(
    "/quran/align-audio",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
)
async def align_quran_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JobAcceptedResponse:
    """
    Accept Quran audio immediately and process it in the background.

    The client receives a job ID and can poll /jobs/{job_id}.
    """

    filename = Path(
        file.filename or "uploaded-recitation"
    ).name

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio or video format.",
        )

    staging_directory = Path(
        tempfile.mkdtemp(prefix="quran-alignment-job-")
    )

    source_path = staging_directory / f"source{extension}"
    maximum_bytes = settings.maximum_upload_mb * 1024 * 1024
    received_bytes = 0

    try:
        with source_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                received_bytes += len(chunk)

                if received_bytes > maximum_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "The uploaded file exceeds the "
                            f"{settings.maximum_upload_mb} MB limit."
                        ),
                    )

                destination.write(chunk)

    except Exception:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )
        raise

    finally:
        await file.close()

    if received_bytes == 0:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded media file is empty.",
        )

    job = job_store.create(
        client_job_id=str(uuid4()),
        project_id="quran-alignment",
        owner_id="api",
        original_filename=filename,
    )

    background_tasks.add_task(
        _process_quran_alignment_job,
        job.id,
        str(source_path),
        filename,
        str(staging_directory),
    )

    return JobAcceptedResponse(
        job_id=job.id,
        client_job_id=job.client_job_id,
        project_id=job.project_id,
        status=job.status,
        created_at=job.created_at,
    )


@app.get(
    "/jobs/{job_id}/result",
    dependencies=[Depends(verify_api_key)],
)
async def get_quran_alignment_result(
    job_id: str,
) -> dict[str, object]:
    """
    Return job progress or the completed Quran alignment result.
    """

    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alignment job not found.",
        )

    if job.status == "completed" and job.result is not None:
        return {
            "job_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "result": job.result,
        }

    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "current_stage": job.current_stage,
        "message": job.message,
        "error": job.error,
        "result": None,
    }

