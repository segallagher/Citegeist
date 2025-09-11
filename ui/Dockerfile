FROM python:3.11-slim

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add app files
COPY static static
COPY templates templates
COPY util util
COPY app.py app.py

CMD ["python3", "app.py"]