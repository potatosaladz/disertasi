import os

from celery import Celery

from worker.celery_tasks import execute_full_shcr_cycle

app = Celery(
    "shcr_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)


@app.task(name="shcr.health_check")
def health_check() -> dict[str, str]:
    return {"status": "SHCR Worker Running"}


@app.task(name="shcr.run_full_shcr_cycle")
def run_full_shcr_cycle_task(scenario_id: int) -> dict[str, object]:
    return execute_full_shcr_cycle(scenario_id)
