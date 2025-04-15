# Use a base image that supports ARM architecture
FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Expose the default MLflow port
EXPOSE 5000

# Set the working directory
WORKDIR /mlflow

# Command to run MLflow (optional: tracking server or UI)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]