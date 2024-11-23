FROM openjdk:11-jdk-slim 
# Install Spark dependencies 
    rm spark-3.1.2-bin-hadoop3.2.tgz 
# Set environment variables 
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 
ENV SPARK_HOME=/opt/spark-3.1.2-bin-hadoop3.2 
ENV PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$PATH 
# Install Python dependencies 
COPY requirements.txt /app/requirements.txt 
RUN pip install -r /app/requirements.txt 
# Set the working directory to the app 
WORKDIR /app 
COPY . /app 
# Run Streamlit app 
CMD ["streamlit", "run", "app.py"] 
