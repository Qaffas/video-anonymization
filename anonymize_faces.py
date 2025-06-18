import cv2
import numpy as np

def load_face_detector(model_file, config_file):
    return cv2.dnn.readNetFromCaffe(config_file, model_file)

def anonymize_faces(frame, net, confidence_threshold=0.3):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            x, y, x2, y2 = max(0, x), max(0, y), min(w, x2), min(h, y2)
            roi = frame[y:y2, x:x2]
            if roi.size == 0:
                continue
            frame[y:y2, x:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
    return frame

def main(input_video, output_video, model_file, config_file, confidence_threshold=0.3):
    net = load_face_detector(model_file, config_file)
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
            frame = anonymize_faces(frame, net, confidence_threshold)
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
    main(
        input_video='input_video.mp4',
        output_video='output_video.mp4',
        model_file='res10_300x300_ssd_iter_140000_fp16.caffemodel',
        config_file='deploy.prototxt',
        confidence_threshold=0.3
    )