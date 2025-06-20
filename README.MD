# Face Anonymization Pipeline (Level 1: The Analyst)

This project implements a real-time face anonymization pipeline for the Waq1 Information Technology Co. interview task (Level 1: The Analyst). It uses OpenCV’s DNN-based ResNet-10 SSD model to detect faces in video or webcam input and applies Gaussian blur to anonymize them, meeting the requirement for real-time processing with minimal dependencies.

## Features
- **Face Detection**: Uses Caffe SSD (ResNet-10) model for robust face detection, with support for Haar Cascade alternatives.
- **Anonymization**: Applies Gaussian blur to detected face regions for privacy.
- **Input Support**: Processes both video files (`inputs/input_video.mp4`) and live webcam feeds.
- **Output**: Saves anonymized video to `outputs/output_video.mp4` and displays real-time results via OpenCV GUI.
- **Dockerized**: Runs in a Docker container for portability, with X11 support for GUI display and webcam access.

## Project Level
This project fulfills **Level 1: The Analyst** requirements, using only OpenCV for face detection and anonymization, ensuring compliance with the task’s constraints.

## Prerequisites
- **Docker**: Installed on the host machine (Linux or Windows).
- **Operating System**:
  - Linux: Preferred for webcam support and X11 forwarding.
  - Windows: Requires VcXsrv for GUI display; webcam support is limited (see notes).
- **Files**:
  - Video file: `inputs/input_video.mp4` (download from https://www.pexels.com/video/people-wearing-face-mask-in-public-area-6960541/).
  - Model files (included in repository):
    - `models/ResNet_10_SSD/deploy.prototxt`
    - `models/ResNet_10_SSD/res10_300x300_ssd_iter_140000_fp16.caffemodel`
    - `models/Haar_Cascade/haarcascade_frontalface_alt_tree.xml`
    - `models/Haar_Cascade/haarcascade_frontalface_alt.xml`
    - `models/Haar_Cascade/haarcascade_frontalface_alt2.xml`
    - `models/Haar_Cascade/haarcascade_frontalface_default.xml`
- **Webcam**: Required for live input (set `INPUT_SOURCE = "webcam"` in `anonymize_faces.py`).

## Project Structure
```
VIDEO-ANONYMIZATION
├── anonymize_faces.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── inputs/
│   └── input_video.mp4
├── models/
│   ├── Haar_Cascade/
│   │   ├── haarcascade_frontalface_alt_tree.xml
│   │   ├── haarcascade_frontalface_alt.xml
│   │   ├── haarcascade_frontalface_alt2.xml
│   │   └── haarcascade_frontalface_default.xml
│   └── ResNet_10_SSD/
│       ├── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000_fp16.caffemodel
└── outputs/
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd VIDEO-ANONYMIZATION
   ```
2. **Download Files** (if not included):
   - Place `input_video.mp4` in `inputs/` (optional if testing with webcam).
   - Model files are included in the repository.
3. **Create Output Directory**:
   ```bash
   mkdir outputs
   ```
4. **Install VcXsrv (Windows Only)**:
   - Download from https://sourceforge.net/projects/vcxsrv/.
   - Run `XLaunch`, select “Multiple windows,” “Start no client,” and check “Disable access control.”

## Build Instructions
Build the Docker image using the provided `Dockerfile`:
```bash
docker build -t face-anonymizer .
```

## Run Instructions
### For Video Input
- **Linux**:
  ```bash
  xhost +local:docker
  docker run --rm -it \
      --env DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $(pwd)/outputs:/app/outputs \
      face-anonymizer
  ```
- **Windows** (with VcXsrv):
  1. Find your IP address:
     ```powershell
     ipconfig
     ```
     Note the “IPv4 Address” (e.g., `192.168.x.x`).
  2. Run:
     ```powershell
     docker run --rm -it `
         --env DISPLAY=192.168.x.x:0 `
         -v Y:\WITCO\VIDEO-ANONYMIZATION\outputs:/app/outputs `
         face-anonymizer
     ```

### For Webcam Input (Linux Only)
- Ensure your webcam is accessible (`/dev/video0`).
- Set `INPUT_SOURCE = "webcam"` in `anonymize_faces.py`.
- Run:
  ```bash
  xhost +local:docker
  docker run --rm -it \
      --env DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $(pwd)/outputs:/app/outputs \
      --device=/dev/video0 \
      face-anonymizer
  ```

### Notes
- **Output**: Processed video is saved to `outputs/output_video.mp4`.
- **GUI Display**: Requires X11 forwarding (Linux) or VcXsrv (Windows). If GUI fails, comment out `cv2.imshow` in `anonymize_faces.py` and check `output_video.mp4`.
- **Webcam on Windows**: Limited support due to Docker’s lack of native webcam passthrough. Use a Linux host or test with video input.
- **Configuration**: Edit `DETECTOR_TYPE` and `INPUT_SOURCE` in `anonymize_faces.py` to switch between Caffe SSD, Haar Cascade, or input sources.

## Troubleshooting
- **GUI Issues**:
  - Ensure `xhost +local:docker` (Linux) or VcXsrv (Windows) is configured.
  - Verify `python3-opencv` is installed in the container (included in `Dockerfile`).
- **Webcam Issues**:
  - Check `/dev/video0` permissions: `ls -l /dev/video*`.
  - Test webcam locally: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`.
- **File Not Found**:
  - Confirm model files and `input_video.mp4` are in the correct paths.
- **Performance**:
  - For faster webcam processing, add resolution limits in `anonymize_faces.py`:
    ```python
    if input_video == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ```

## Submission
This project is submitted as part of the Waq1 Information Technology Co. interview for the Data Scientist position (Level 1: The Analyst). The repository is public and contains all necessary files and instructions.