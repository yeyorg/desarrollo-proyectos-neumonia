#recomended from uv docs
FROM python:3.12-slim-trixie

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the project files
COPY README.md ./
COPY detector-neumonia-uv/pyproject.toml ./detector-neumonia-uv/
COPY detector-neumonia-uv/src ./detector-neumonia-uv/src

# Copy the model file
COPY detector-neumonia-uv/models/conv_MLP_84.h5 ./detector-neumonia-uv/models/

# Create directory for medical images
RUN mkdir -p /app/imagenes

# Define volume for medical images
VOLUME ["/app/imagenes"]

# Install dependencies using UV
WORKDIR /app/detector-neumonia-uv
RUN uv sync --no-dev

# Run the console app from the project root
CMD ["uv", "run", "python", "src/main.py", "--console"]
