FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for reportlab and some scientific packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (we'll generate specific ones or use existing)
# Creating a specific requirements file for docker to avoid issues
COPY requirements.txt .
# Add new dependencies if not in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sentence-transformers networkx duckduckgo-search mcp pymongo chromadb reportlab

COPY . .

CMD ["python", "app.py"]
