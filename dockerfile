FROM python:3.10-slim-bullseye

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p final_model

# Expose the port your Flask app runs on
EXPOSE 8080

# Run the Flask application
CMD ["python3", "app.py"]