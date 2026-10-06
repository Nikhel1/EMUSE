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

# Copy application files.
# The three large data files (epoch_99.pt, all_sbid_image_features.pt,
# allidx_sbid_ra_dec_flux_catwise.pkl) are downloaded by the CI smoke-test
# job and passed to this build via the Actions cache, so they arrive in the
# build context and are copied in here — no runtime download needed.
COPY . .

# Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start app
CMD ["streamlit", "run", "main.py"]
