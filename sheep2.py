import cv2
from ultralytics import YOLO

# Colors
violet = (234, 94, 166)
yellow = (48, 211, 254)
green = (129, 222, 38)
orange = (0, 127, 255)

tracked_objects = {}
counted_sheep = 0

region1=[10,600,1880,1050]

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
    return iou > 0.80  # True if overlap > 98%

# Load the YOLO model
model = YOLO("/Users/mst.sadiakhatun/Desktop/Sheep_counter/Model_weight/yolo11m.pt")

# Open the video file
video_path = "/Users/mst.sadiakhatun/Desktop/Sheep_counter/Sheep_video/sheep_counter_video.mp4"
cap = cv2.VideoCapture(video_path)

# Define output video settings
output_path = "/Users/mst.sadiakhatun/Desktop/Sheep_counter/Sheep_video/output/sheep_count_video6.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.70, device='mps',tracker="/Users/mst.sadiakhatun/Desktop/Sheep_counter/Sheep_video/bytetrack.yaml")
        
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name == 'sheep':
                    bbox = box.xyxy.cpu().numpy()[0]
                    track_id = box.id.item()  # Get the track ID
                    x1, y1, x2, y2 = map(int, bbox)

                    # Check if the sheep is already tracked and counted
                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = {
                            'counted': False,
                            'bbox': bbox
                        }

                    # Check if the sheep is overlapping and not yet counted
                    if not tracked_objects[track_id]['counted']:
                        for other_id, data in tracked_objects.items():
                            if other_id != track_id and is_overlapping(bbox, region1):
                                tracked_objects[track_id]['counted'] = True
                                counted_sheep += 1
                                break

                    # Update bounding box color and text based on counting status
                    if tracked_objects[track_id]['counted']:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), orange, thickness=2)
                        cv2.putText(frame, f"ID: {track_id} | Counted", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, orange, 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), green, thickness=2)
                        cv2.putText(frame, f"ID: {track_id} | {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)

        # Overlay total sheep count on the frame
        cv2.rectangle(frame, (40, 10) ,(650, 100), violet, thickness=-1)
        cv2.putText(frame, f"Total Counted Sheep: {counted_sheep}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

        # Write the frame to the output video
       

        # Display the frame
        overlay = frame.copy()
        cv2.rectangle(frame, (10, 600), (1880, 1050), violet, thickness=2)
        cv2.rectangle(overlay, (10, 600), (1880, 1050), violet, thickness=-1)
        cv2.addWeighted(overlay, 0.3, frame, 1 - 0.3, 0, frame)
        out.write(frame)
        cv2.imshow("YOLO Sheep Counting", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture and writer objects, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
