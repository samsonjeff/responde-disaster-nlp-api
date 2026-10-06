# Python 3.11 slim for optimal balance of container size and wheel compatibility
FROM python:3.11-slim

# Set environment variables for Python runtime & default port
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create non-root user (UID 1000) for security & cloud container compatibility
RUN useradd -m -u 1000 user
USER user

WORKDIR $HOME/app

# Pre-install CPU-only PyTorch to avoid massive CUDA wheels (~2GB saved)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy application code and model weights
COPY --chown=user . .

# Expose default HTTP port
EXPOSE 8000

# Start FastAPI server binding to $PORT with reverse-proxy header support
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]