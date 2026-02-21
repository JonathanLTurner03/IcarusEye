# IcarusEye v2 — Tower service (dashboard + MJPEG server)
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt  /tmp/requirements-base.txt
COPY requirements/tower.txt /tmp/requirements-tower.txt
RUN pip install --no-cache-dir \
    -r /tmp/requirements-base.txt \
    -r /tmp/requirements-tower.txt

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

COPY pipeline/ ./pipeline/

CMD ["uvicorn", "pipeline.tower.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]