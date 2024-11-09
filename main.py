import cv2
from ultralytics import YOLO
import easyocr
import datetime
 
# Capture video file into OpenCV for image processing
cap = cv2.VideoCapture(r'C:\Users\LAB203_xx\Desktop\ml-license-plate\traffic-highway.mp4')

# Initialize YOLO model
model = YOLO(r'C:\Users\LAB203_xx\Desktop\ml-license-plate\license-plate-final-best.pt')

# Initialize EasyOCR. Supports both English and Thai.
reader = easyocr.Reader(['en'])

fps = cap.get(cv2.CAP_PROP_FPS)

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

def run_ocr(plate):
    """
    Run the Optical Character Recognition on the given plate image. Return recognized text.
    """
    try:
        cv2.imshow("plate", plate)
        results = reader.readtext(plate)
        final_text = []
        for result in results:
            _, text = result
            final_text.append(text)
        
        final_text = " ".join(final_text)
        return final_text
    except:
        return None

def process_frame():
    """
    Function that process each frame of the video. If the program should be terminated, return True.
    """

    # Calculate the start time of processing the frame
    start = datetime.datetime.now()

    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return True

    # Run YOLO with tracking enabled
    results = model.track(frame, conf=CONFIDENCE_THRESHOLD, persist=True, tracker="botsort.yaml")[0]
    
    # Process tracked detections
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        
        # Draw boxes and process each detection
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get the license plate region
            plate = frame[y1:y2, x1:x2, :]

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), GREEN, -1)
            cv2.putText(frame, str(track_id), (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            
            # Run OCR on the plate
            text = run_ocr(plate)
            if text:
                cv2.putText(frame, text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # Calculate and display FPS
    end = datetime.datetime.now()
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Show output frame.
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        return True
    
while cap.isOpened():
    if process_frame():
        break

cap.release()
cv2.destroyAllWindows()