# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONUNBUFFERED=1

# Set a working directory inside the image
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . /app/

# Expose Gradio default port
EXPOSE 7860

# Gradio listens on all interfaces inside the container
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Default command launches the RAG Gradio interface (watcher started via compose)
CMD ["python", "app.py"] 