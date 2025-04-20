import cv2

from ultralytics import YOLO

# violet =(234, 94, 166)
# yellow =(48, 211, 254)
# green =(129, 222, 38)
# orange = (0, 127, 255)


tracked_objects = {}
def is_overlapping(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return False  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    iou = intersection_area / bbox1_area
    return iou > 0.98  # True if overlap > 98%


# Load the YOLO11 model
model = YOLO("/Users/mst.sadiakhatun/Desktop/Sheep_counter/Model_weight/yolo11m.pt")

# Open the video file
video_path = "/Users/mst.sadiakhatun/Desktop/Sheep_counter/Sheep_video/sheep_counter_video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.60, device='mps')
        
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name == 'sheep':
                    bbox = box.xyxy.cpu().numpy()[0]
                    track_id = box.id.item()  # Get the track ID
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (129, 222, 38), thickness=2)
                    cv2.putText(frame, f"ID: {track_id} | {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (129, 222, 38), 2)
                    
                    if track_id not in tracked_objects:
                        # New car, set as "Arrived" and start timing
                        tracked_objects[track_id] = {
                            'counted': False,
                        }
                        
                    
        

        overlay = frame.copy()

                    # Display the annotated frame
        cv2.rectangle(frame, (10, 600), (1880, 1050), (234, 94, 166), thickness=2)
        cv2.rectangle(overlay, (10, 600), (1880, 1050), (234, 94, 166), thickness=-1)
        cv2.addWeighted(overlay, 0.3, frame, 1 -0.3, 0, frame)
            
        cv2.imshow("YOLO11 Tracking", frame)
                    

                    # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()