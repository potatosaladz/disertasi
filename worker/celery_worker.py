import os
from typing import Any

from celery import Celery

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
def run_full_shcr_cycle_task(self: Any, scenario_id: int) -> dict[str, object]:
    def report(logs: list[str]) -> None:
        self.update_state(state="PROGRESS", meta={"logs": logs})

    return execute_full_shcr_cycle(scenario_id, progress=report)
