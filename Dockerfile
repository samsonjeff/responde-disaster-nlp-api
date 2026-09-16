# Use Python 3.11 slim — good balance of size and library support
FROM python:3.11-slim

# HF Spaces runs as a non-root user; create a working directory
WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
