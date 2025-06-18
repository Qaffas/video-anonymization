import cv2
import numpy as np

# Load the pre-trained DNN face detection model
model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Open the video file
input_video = 'input_video.mp4'
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for DNN face detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    # Process each detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)
            roi = frame[y:y2, x:x2]
            if roi.size == 0:
                continue
            roi = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[y:y2, x:x2] = roi

    # Display and save the frame
    cv2.imshow('Anonymized Video', frame)
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()