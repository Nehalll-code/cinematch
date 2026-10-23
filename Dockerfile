FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
COPY artifacts/ artifacts/
ENV CINEMATCH_MODEL=artifacts/model.pkl PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "cinematch.api:app", "--host", "0.0.0.0", "--port", "8000"]
