import os
from datetime import datetime, timezone
from typing import Any

from celery import Celery

from backend.database import SessionLocal
from backend.models import ConsensusSession
from worker.celery_tasks import execute_full_shcr_cycle

app = Celery(
    "shcr_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)
app.conf.task_track_started = True


@app.task(name="shcr.health_check")
def health_check() -> dict[str, str]:
    return {"status": "SHCR Worker Running"}


@app.task(bind=True, name="shcr.run_full_shcr_cycle")
def run_full_shcr_cycle_task(
    self: Any,
    scenario_id: int,
    session_id: str,
) -> dict[str, object]:
    def report(logs: list[dict[str, str]]) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"logs": logs, "scenario_id": scenario_id, "session_id": session_id},
        )

    try:
        return execute_full_shcr_cycle(scenario_id, session_id, progress=report)
    except Exception as error:
        with SessionLocal() as session:
            run = session.get(ConsensusSession, session_id)
            if run is not None and run.scenario_id == scenario_id:
                run.status = "FAILED"
                run.error = f"{type(error).__name__}: {error}"
                run.completed_at = datetime.now(timezone.utc)
                session.commit()
        raise
