from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AlignmentJob:
    client_job_id: str
    project_id: str
    owner_id: str
    original_filename: str

    id: str = field(
        default_factory=lambda: str(uuid4())
    )

    status: str = "engine_not_ready"
    progress: int = 0
    current_stage: str = "engine_not_ready"

    message: str = (
        "The API is connected, but the Quran model "
        "and verified corpus are not loaded yet."
    )

    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, object] | None = None

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, AlignmentJob] = {}
        self._lock = Lock()

    def create(
        self,
        *,
        client_job_id: str,
        project_id: str,
        owner_id: str,
        original_filename: str,
    ) -> AlignmentJob:
        job = AlignmentJob(
            client_job_id=client_job_id,
            project_id=project_id,
            owner_id=owner_id,
            original_filename=original_filename,
        )

        with self._lock:
            self._jobs[job.id] = job

        return job

    def get(self, job_id: str) -> AlignmentJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job_id: str) -> AlignmentJob | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if job.status == "cancelled":
                return job

            job.status = "processing"
            job.progress = 5
            job.current_stage = "starting"
            job.message = "The Quran alignment job has started."
            job.error = None
            job.updated_at = utc_now()

            return job

    def update(
        self,
        job_id: str,
        *,
        progress: int,
        current_stage: str,
        message: str,
    ) -> AlignmentJob | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if job.status == "cancelled":
                return job

            job.status = "processing"
            job.progress = max(0, min(progress, 99))
            job.current_stage = current_stage
            job.message = message
            job.updated_at = utc_now()

            return job

    def complete(
        self,
        job_id: str,
        result: dict[str, object],
    ) -> AlignmentJob | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if job.status == "cancelled":
                return job

            job.status = "completed"
            job.progress = 100
            job.current_stage = "completed"
            job.message = "The Quran alignment job completed successfully."
            job.result = result
            job.error = None
            job.updated_at = utc_now()

            return job

    def fail(
        self,
        job_id: str,
        error: str,
    ) -> AlignmentJob | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if job.status == "cancelled":
                return job

            job.status = "failed"
            job.current_stage = "failed"
            job.message = "The Quran alignment job failed."
            job.error = error
            job.updated_at = utc_now()

            return job

    def cancel(self, job_id: str) -> AlignmentJob | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            job.status = "cancelled"
            job.current_stage = "cancelled"
            job.message = "The alignment job was cancelled."
            job.updated_at = utc_now()

            return job


job_store = InMemoryJobStore()