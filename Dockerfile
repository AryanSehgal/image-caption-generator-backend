FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ResNet50's ImageNet weights during build, so the app
# doesn't need to fetch them from the internet every time it starts
RUN python -c "from tensorflow.keras.applications.resnet50 import ResNet50; ResNet50(weights='imagenet')"

COPY . .

CMD gunicorn -b 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 wsgi:app