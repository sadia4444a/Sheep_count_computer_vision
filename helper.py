import cv2
import os

# Function to extract frames from a video
def extract_frames(video_path, frame_count, output_folder):
    """
    Extracts the specified number of frames from a video and saves them to the output folder.

    :param video_path: Path to the video file
    :param frame_count: Number of frames to extract
    :param output_folder: Folder to save the extracted frames
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count > total_frames:
        frame_count = total_frames

    frame_interval = total_frames // frame_count

    count = 0
    saved_frames = 0

    while cap.isOpened() and saved_frames < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frames + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        count += 1

    cap.release()

# Function to get coordinate values for drawing shapes
def get_coordinate_value(image_path, shape_type):
    """
    Opens an image and lets the user select coordinates for drawing shapes.

    :param image_path: Path to the image file
    :param shape_type: Type of shape ('line' or 'rectangle')
    :return: Coordinates for the specified shape
    """
    def click_event(event, x, y, flags, params):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if (shape_type == 'line' and len(points) == 2) or (shape_type == 'rectangle' and len(points) == 2):
                cv2.destroyAllWindows()

    points = []
    image = cv2.imread(image_path)

    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", click_event)
    cv2.waitKey(0)

    if shape_type == 'line':
        return points[:2]  # Two points for the line
    elif shape_type == 'rectangle':
        top_left = points[0]
        bottom_right = points[1]
        return top_left, bottom_right
    else:
        raise ValueError("Unsupported shape type. Use 'line' or 'rectangle'.")
