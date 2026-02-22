# IcarusEye v2 — Pipeline service (lens + hawk + mark)
FROM dustynv/l4t-ml:r36.4.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt  /tmp/requirements-base.txt
COPY requirements/lens.txt  /tmp/requirements-lens.txt
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    -r /tmp/requirements-base.txt \
    -r /tmp/requirements-lens.txt

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY pipeline/ ./pipeline/

CMD ["python3", "-m", "pipeline.pipeline_service.main"]