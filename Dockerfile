FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --no-cache-dir -e .

COPY . .

CMD ["python", "-m", "src.storage.chroma_service"]
