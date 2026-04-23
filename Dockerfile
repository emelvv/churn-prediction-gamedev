FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY config ./config
COPY data_pipeline ./data_pipeline
COPY training ./training
COPY inference ./inference
COPY api ./api
COPY pipelines ./pipelines
COPY data ./data
COPY artifacts ./artifacts
COPY README.md ./README.md

RUN python pipelines/train_pipeline.py

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
