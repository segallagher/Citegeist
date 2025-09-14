FROM python:3.11-slim

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add ui files
COPY ui.static static
COPY ui.templates templates
COPY ui.app.py app.py

# Add rag files
COPY rag rag

# Add dataset creation files
COPY dataset_creation dataset_creation

CMD ["python3", "app.py"]