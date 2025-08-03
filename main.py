import cv2
import torch
import numpy as np
import os

# Load YOLOv5 pretrained model from torch hub
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # Confidence threshold

def mock_segmentation(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented

def detect_cracks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def overlay_results(frame, detections, cracks, segment):
    for *xyxy, conf, cls in detections:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    small_segment = cv2.resize(segment, (160, 120))
    frame[10:130, 10:170] = small_segment

    frame = cv2.addWeighted(frame, 1, cracks, 0.5, 0)
    return frame

def process_video(video_path, output_path='output.mp4'):
    if not os.path.exists(video_path):
        print("Video not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Use mp4v codec for mp4 output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Processing video... Press 'q' to quit.")

    cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)  # Optional: resizable window

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        segment = mock_segmentation(frame)
        cracks = detect_cracks(frame)
        overlayed = overlay_results(frame.copy(), detections, cracks, segment)

        out.write(overlayed)
        cv2.imshow("Processed Frame", overlayed)

        if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms delay to see frames clearly
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Output saved to", output_path)

if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(__file__), "icehockey.mp4")
    process_video(video_path, output_path='output.mp4')
