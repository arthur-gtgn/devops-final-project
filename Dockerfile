FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Ensure Python modules in /app are importable
ENV PYTHONPATH=/app

# Start both FastAPI and Streamlit on different ports
CMD ["sh", "-c", "uvicorn src.mushroom_ml.api:app --host 0.0.0.0 --port 8000 & streamlit run src/mushroom_ml/app.py --server.port 8501 --server.address 0.0.0.0"]
