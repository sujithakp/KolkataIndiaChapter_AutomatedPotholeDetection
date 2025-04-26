FROM python:3.9-slim
WORKDIR /app

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY services/ services/
COPY tasks/ tasks/
COPY utils/ utils/
COPY api/ api/
COPY setup_cloudinary.py .
COPY config.py .
COPY config.py .


# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (overridden for Celery in docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

