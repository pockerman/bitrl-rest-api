# ================================
#  FastAPI + Gymnasium Environment API
#  Using uv for dependency management
# ================================
FROM python:3.12-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv (https://docs.astral.sh/uv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy the pyproject.toml and optional uv.lock first
COPY pyproject.toml uv.lock* /app/


# Install project dependencies (production only)
RUN uv sync --no-dev --frozen

# Copy the application source
COPY . /app

# Expose FastAPI port
EXPOSE 8001


# Start FastAPI server with uvicorn
CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8001"]
