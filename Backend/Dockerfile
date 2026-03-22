FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system deps required by gitpython and common ML wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY Backend/requirements_api.txt /app/Backend/requirements_api.txt
RUN pip install --upgrade pip && pip install -r /app/Backend/requirements_api.txt

COPY Backend /app/Backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "Backend.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
