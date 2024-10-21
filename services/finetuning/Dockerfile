FROM python:3.11

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY finetune.py .

#EXPOSE 8080

CMD ["python", "finetune.py"]