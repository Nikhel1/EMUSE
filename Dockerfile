FROM python:3.11-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (removed in same layer to keep image small)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies - single layer, no cache
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy only the necessary application files (see .dockerignore for exclusions)
# Large data files (epoch_99.pt, all_sbid_image_features.pt, allidx_sbid_ra_dec_flux_catwise.pkl,
# clip_pretrained/) are excluded - they are downloaded at runtime via gdown / open_clip
COPY . .

# Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start app
CMD ["streamlit", "run", "main.py"]
