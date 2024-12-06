FROM python:3.13.0-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY dataset-download.py .

CMD ["python", "dataset-download.py"]
