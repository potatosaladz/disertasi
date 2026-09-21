FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY worker/requirements.txt ./worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt

COPY backend ./backend
COPY worker ./worker

RUN useradd --create-home --uid 10001 appuser
USER appuser

CMD ["celery", "-A", "worker.celery_worker:app", "worker", "--loglevel=info"]
