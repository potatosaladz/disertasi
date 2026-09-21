import os

from celery import Celery

app = Celery(
    "shcr_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)


@app.task(name="shcr.health_check")
def health_check() -> dict[str, str]:
    return {"status": "SHCR Worker Running"}
