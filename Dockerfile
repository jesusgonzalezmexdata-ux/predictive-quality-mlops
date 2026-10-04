FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
COPY models ./models
COPY reports ./reports
COPY app ./app
ENV PYTHONPATH=/app/src PQM_DB=/data/predicciones.db PQM_MODEL_DIR=/app/models
VOLUME /data
EXPOSE 8000 8501
CMD ["uvicorn", "pqm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
