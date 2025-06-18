import cv2
import numpy as np

class FaceDetector:
    def detect(self, frame):
        """Detect faces in a frame. Returns list of (x, y, x2, y2) boxes."""
        raise NotImplementedError

class CaffeSSDDetector(FaceDetector):
    def __init__(self, model_file, config_file, confidence_threshold=0.3):
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x, y, x2, y2 = max(0, x), max(0, y), min(w, x2), min(h, y2)
                boxes.append((x, y, x2, y2))
        return boxes

class HaarCascadeDetector(FaceDetector):
    def __init__(self, cascade_file):
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append((x, y, x + w, y + h))
        return boxes

def anonymize_faces(frame, detector: FaceDetector):
    boxes = detector.detect(frame)
    for (x, y, x2, y2) in boxes:
        # Calculate center and axes for the ellipse
        center_x = (x + x2) // 2
        center_y = (y + y2) // 2
        axis_x = (x2 - x) // 2  # Half width
        axis_y = (y2 - y) // 2  # Half height, adjust for more oval shape if needed
        # Adjust axis_y to make it slightly longer for an egg-like shape
        axis_y = int(axis_y * 1.2)  # Increase vertical axis by 20% for oval effect

        # Create a mask for the ellipse
        mask = np.zeros_like(frame)
        cv2.ellipse(
            mask,
            center=(center_x, center_y),
            axes=(axis_x, axis_y),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=(255, 255, 255),
            thickness=-1  # Filled ellipse
        )

        # Blur the region of interest
        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            continue
        blurred_roi = cv2.GaussianBlur(frame, (99, 99), 30)

        # Apply the mask to blend the blurred region
        masked_blur = np.where(mask != 0, blurred_roi, frame)
        frame = masked_blur

    return frame

def main(input_video, output_video, detector: FaceDetector):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = anonymize_faces(frame, detector)
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
            cv2.imshow('Anonymized Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Processing complete.")

if __name__ == "__main__":
    
    #>>>>>>>>>>>>> Configure mnodel and input here <<<<<<<<<<<<<
    
    # Change to selected model type:
    # ["caffe", "haar1", "haar2", "haar3", "haar4"]
    DETECTOR_TYPE = "caffe"
    
    # Change to input source:
    # "webcam" for live webcam feed or path to video file
    #INPUT_SOURCE = "inputs/input_video.mp4"
    INPUT_SOURCE = "webcam" 
    
    ############################################################



    if DETECTOR_TYPE == "caffe":
        detector = CaffeSSDDetector(
            model_file='models/ResNet_10_SSD/res10_300x300_ssd_iter_140000_fp16.caffemodel',
            config_file='models/ResNet_10_SSD/deploy.prototxt',
            confidence_threshold=0.25
        )
    elif DETECTOR_TYPE == "haar1":
        detector = HaarCascadeDetector('models/Haar_Cascade/haarcascade_frontalface_default.xml')

    elif DETECTOR_TYPE == "haar2":
        detector = HaarCascadeDetector('models/Haar_Cascade/haarcascade_frontalface_alt.xml')

    elif DETECTOR_TYPE == "haar3":
        detector = HaarCascadeDetector('models/Haar_Cascade/haarcascade_frontalface_alt2.xml')

    elif DETECTOR_TYPE == "haar4":
        detector = HaarCascadeDetector('models/Haar_Cascade/haarcascade_frontalface_alt_tree.xml')


    input_source = 0 if INPUT_SOURCE == "webcam" else INPUT_SOURCE

    main(
        input_video=input_source,
        output_video='outputs/output_video.mp4',
        detector=detector
    )
