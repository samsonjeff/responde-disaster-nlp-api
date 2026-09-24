# Use Python 3.11 slim - good balance of size and library support
FROM python:3.11-slim

# Create non-root user (UID 1000) for security & cloud container compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies with CPU-only PyTorch for fast builds and small image size
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code and model weights
COPY --chown=user . .

# Expose port (Cloud Run sets $PORT dynamically, default to 7860)
EXPOSE 7860 8080

# Start FastAPI server binding to $PORT dynamically
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]