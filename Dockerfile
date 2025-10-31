FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create persistent upload directory
RUN mkdir -p /app/uploaded_papers

EXPOSE 8000

ENV UPLOAD_DIR=/app/uploaded_papers \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/research_papers

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


