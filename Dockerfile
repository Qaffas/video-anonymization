FROM python:3.10-slim

# Install system dependencies for GUI and webcam
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libv4l-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Python OpenCV
RUN pip install --no-cache-dir opencv-python

# Set working directory
WORKDIR /app

# Copy project files
COPY anonymize_faces.py .
COPY models/ResNet_10_SSD/ models/ResNet_10_SSD/
COPY models/Haar_Cascade/ models/Haar_Cascade/
COPY inputs/input_video.mp4 inputs/
COPY .dockerignore .
COPY .gitignore .

# Create outputs directory
RUN mkdir -p outputs

# Set environment variable for GUI display
ENV DISPLAY=:0

# Command to run the script
CMD ["python", "anonymize_faces.py"]