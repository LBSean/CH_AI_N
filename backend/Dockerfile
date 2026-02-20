FROM python:3.11-slim

WORKDIR /app

# System deps required for psycopg binary and compiling Python extensions
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast pip replacement) then install project deps
COPY pyproject.toml .
RUN pip install uv && uv pip install --system -e .

# Copy application source
COPY . .

EXPOSE 8000

# Production: 2 workers; override via docker-compose for dev (--reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
