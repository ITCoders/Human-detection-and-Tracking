import torch
import cv2
import torchvision.transforms as T
from torchvision.models.detection import ssd300_vgg16
import numpy as np
import time  # Added for time tracking

# Load the pre-trained SSD model with VGG16 backbone
model = ssd300_vgg16(pretrained=False)
model.load_state_dict(torch.load('ssd300_vgg16_pretrained.pth'))
model.eval()
print("Model weights loaded successfully.")

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a transform to resize the frame for model input
input_size = (300, 300)  # SSD300 expects 300x300 input
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(input_size),
    T.ToTensor()
])

# Function to get detections
def get_detections(frame, original_width, original_height, threshold=0.3):
    frame_transformed = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(frame_transformed)[0]

    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    high_conf_indices = scores > threshold
    boxes = boxes[high_conf_indices]

    rescaled_boxes = []
    for box in boxes:
        xmin = int(box[0] * original_width / input_size[0])
        ymin = int(box[1] * original_height / input_size[1])
        xmax = int(box[2] * original_width / input_size[0])
        ymax = int(box[3] * original_height / input_size[1])
        rescaled_boxes.append((xmin, ymin, xmax, ymax))

    return rescaled_boxes, scores[high_conf_indices]

# MultiTracker setup
multi_tracker = cv2.legacy.MultiTracker_create()
tracking = False
detected_boxes = []  # Store detected bounding boxes
tracking_data = {}  # Stores entry time of tracked objects

# Mouse callback to select bounding boxes for tracking
def select_bbox(event, x, y, flags, param):
    global tracking, detected_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in detected_boxes:
            x_min, y_min, x_max, y_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                tracker = cv2.legacy.TrackerCSRT_create()
                multi_tracker.add(tracker, frame, selected_box)
                tracking_data[selected_box] = {"start_time": time.time()}  # Initialize time tracking
                tracking = True
                break

# Set up video capture
video_path = 'myvideo1.mp4'
cap = cv2.VideoCapture(video_path)
cv2.namedWindow("Video")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 4 != 0:  # Process every 4th frame
        frame_count += 1
        continue

    original_height, original_width = frame.shape[:2]

    # Update tracking
    active_boxes = []
    if tracking:
        success, tracked_boxes = multi_tracker.update(frame)
        for box in tracked_boxes:
            x, y, w, h = [int(v) for v in box]
            current_time = time.time()
            box_tuple = (x, y, x + w, y + h)

            # Check if object is still within screen & time limit
            if (
                x > 0 and y > 0 and (x + w) < original_width and (y + h) < original_height
                and (current_time - tracking_data.get(box_tuple, {"start_time": current_time})["start_time"]) < 12
            ):
                active_boxes.append(box_tuple)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Purple for tracking
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Remove objects that have exceeded 12 sec or exited screen
    tracking_data = {box: tracking_data.get(box, {"start_time": time.time()}) for box in active_boxes}

    # Perform detection for new objects
    if not tracking or frame_count % 12 == 0:  # Refresh detections periodically
        detected_boxes, scores = get_detections(frame, original_width, original_height, threshold=0.3)
        for box in detected_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green for detection
            cv2.putText(frame, "Detection", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Reset mouse callback to select new boxes
        cv2.setMouseCallback("Video", select_bbox, detected_boxes)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()