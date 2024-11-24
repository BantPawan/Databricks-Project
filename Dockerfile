FROM python:3.9-slim

# Install necessary dependencies for Spark
RUN apt-get update && apt-get install -y openjdk-11-jdk wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Java environment variables for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
