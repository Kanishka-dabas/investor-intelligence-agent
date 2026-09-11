FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Force CPU-only torch — avoids pulling ~2GB+ of unnecessary CUDA/GPU packages
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY evaluation/ ./evaluation/

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]