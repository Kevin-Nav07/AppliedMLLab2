FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY wsgi.py .
COPY .env.example .
COPY models/checkpoints/best_deeplabv3_building.pt ./models/checkpoints/best_deeplabv3_building.pt

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "wsgi:app"]