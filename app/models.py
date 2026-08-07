from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal[
    "queued",
    "engine_not_ready",
    "processing",
    "extracting_audio",
    "transcribing",
    "searching_quran",
    "matching_passage",
    "aligning_words",
    "preparing_results",
    "review_required",
    "completed",
    "failed",
    "cancelled",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    status: Literal[
        "healthy",
        "degraded",
        "not_ready",
    ]
    engine_version: str
    corpus_loaded: bool
    model_loaded: bool
    message: str


class JobAcceptedResponse(BaseModel):
    job_id: str
    client_job_id: str
    project_id: str
    status: JobStatus
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    client_job_id: str
    project_id: str
    owner_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    current_stage: str
    message: str
    warnings: list[str]
    error: str | None
    created_at: datetime
    updated_at: datetime


class CancelJobResponse(BaseModel):
    job_id: str
    status: Literal["cancelled"]
    message: str