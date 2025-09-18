FROM python:3.11-slim

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add ui files
COPY app/static static
COPY app/templates templates
COPY app/app.py app.py

# Add rag files
COPY app/rag rag

# Add dataset creation files
COPY dataset_creation dataset_creation

CMD ["python3", "app.py"]