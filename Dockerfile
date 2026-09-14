FROM python:3.11-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies + AWS CLI v2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    git \
    build-essential \
    curl \
    unzip \
    groff \
    less \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    && aws --version \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies - single layer, no cache
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Download large model/data files at build time so they are baked into the
# image and never fetched at container start-up.
# Each file is a separate RUN layer so Docker cache can reuse them
# independently if only one changes.
RUN python -m gdown "https://drive.google.com/uc?id=1k0MNw1hyBDejxOovKwhQCPRmJil13ut5" \
      -O /app/epoch_99.pt

RUN python -m gdown "https://drive.google.com/uc?id=11l-iVak_8QnycuePIvPwXbDBUP_ILP_Y" \
      -O /app/all_sbid_image_features.pt

RUN python -m gdown "https://drive.google.com/uc?id=1rI1RzKDMMKrOyeE_7BaCNthYrYgYoRf8" \
      -O /app/allidx_sbid_ra_dec_flux_catwise.pkl

# Copy application files (*.pt and *.pkl excluded via .dockerignore —
# they are already present from the RUN layers above)
COPY . .

# Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start app
CMD ["streamlit", "run", "main.py"]
