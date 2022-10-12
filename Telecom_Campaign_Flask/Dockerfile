# Create the base image
FROM python:3.7-slim

# Change the working directory
WORKDIR /app/

# Copy local folder into the container
COPY . /app/

# Install Dependency
RUN pip install -r requirements.txt

# Set the command as the script name
ENTRYPOINT ["python"]
CMD ["api.py"]

# Expose port 5000
EXPOSE 5000
