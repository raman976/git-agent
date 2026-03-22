FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system deps required by gitpython and common ML wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY Backend /app/Backend

EXPOSE 8000

CMD ["sh", "-c", "ulimit -n 65535 && exec python -m uvicorn Backend.api_server:app --host 0.0.0.0 --port 8000"]
