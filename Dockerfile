# Theo Multi-Agent Dockerfile
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip and poetry (for pyproject.toml support)
RUN pip install --upgrade pip poetry

# Copy pyproject.toml and poetry.lock if present
COPY pyproject.toml ./
# COPY poetry.lock ./  # Uncomment if you have a poetry.lock

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the code
COPY . .

# Copy .env file (if present)
COPY .env .env

# Make entrypoint script executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy test_llm.py
COPY test_llm.py /app/test_llm.py

# Expose the FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"] 