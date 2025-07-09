# Start from a slim Python base
FROM python:3.11-slim

# Install OS build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libopenblas-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Pre-install numpy and cython before copying to prevent repeated rebuilds
RUN pip install --upgrade pip
RUN pip install numpy Cython

# Copy the whole project into the container
COPY . /app

# Install the package (cython extensions + dependencies)
RUN pip install -e .

# Optional: run tests or verify build
# RUN pytest

# Default entrypoint
CMD ["python"]
