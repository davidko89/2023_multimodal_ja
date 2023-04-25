import csv 
import cv2
import dlib
import numpy as np
from pathlib import Path

def rotation_vector_to_euler_angles(rotation_vector):
    """Convert a rotation vector to Euler angles (yaw, pitch, and roll)."""
    matrix, _ = cv2.Rodrigues(rotation_vector)
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((matrix, np.zeros((3, 1)))))
    return euler_angles

PROJECT_PATH = Path(__file__).parents[1]
RAW_DATA_PATH = Path(PROJECT_PATH, "data/raw_data")
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")
HEADPOSE_PATH = Path(PROJECT_PATH, "data/headpose_data")
FACE_LANDMARKS_PATH = Path(PROJECT_PATH, "code/src/shape_predictor_68_face_landmarks.dat")


# Create a Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(Path(FACE_LANDMARKS_PATH)))


# Camera intrinsic parameters (you should use the actual values for your camera)
fx, fy = 961.8638, 550.6279  # focal length in pixels
cx, cy = 912.718, 912.5225  # image center
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# Define 3D model points without depth
model_points_without_depth = np.array([
    (0.0, 0.0, 0.0),    # Nose tip
    (-30.0, -20.0, -30.0),   # Left eye
    (30.0, -20.0, -30.0),    # Right eye
    (0.0, 50.0, -50.0),   # Mouth center
    (-20.0, 50.0, -50.0),   # Left mouth corner
    (20.0, 50.0, -50.0),    # Right mouth corner
], dtype=np.float32)


# Iterate through each depth and RGB file
for color_file in Path(PROC_DATA_PATH, "rgb").glob("*.npy"):
    # Construct the corresponding depth file path
    depth_file = Path(PROC_DATA_PATH, "depth", color_file.stem.replace("color", "depth") + ".npy")

    # Check if the depth file exists before processing the color image
    if depth_file.exists():
        # Load the depth and RGB images as numpy arrays
        depth_image = np.load(depth_file)
        color_image = np.load(color_file)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Get the participant ID from the file name
        participant_id = color_file.stem.split("_")[0]

        # Detect faces in the color image
        faces = detector(color_image, 0)
        print("Number of detected faces:", len(faces))

        # Draw bounding boxes around the detected faces
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Iterate through each detected face
        for face in faces:
            
            # Get the landmarks for the face
            landmarks = predictor(color_image, face)

            # Extract the relevant landmarks for head pose estimation (2D image points)
            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
                ((landmarks.part(48).x + landmarks.part(54).x) // 2, (landmarks.part(48).y + landmarks.part(54).y) // 2),  # Mouth center
                (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
                (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
            ], dtype=np.float32)

            # Get depth values for the landmarks
            depths = [
                depth_image[landmarks.part(30).y, landmarks.part(30).x],
                depth_image[landmarks.part(36).y, landmarks.part(36).x],
                depth_image[landmarks.part(45).y, landmarks.part(45).x],
                depth_image[(landmarks.part(48).y + landmarks.part(54).y) // 2, (landmarks.part(48).x + landmarks.part(54).x) // 2],
                depth_image[landmarks.part(48).y, landmarks.part(48).x],
                depth_image[landmarks.part(54).y, landmarks.part(54).x],
            ]

            # Create 3D model points using depth information
            model_points = model_points_without_depth.copy()
            model_points[:, 2] += depths

            # Solve the PnP problem
            _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)

            # Estimate head pose using solvePnP
            _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)

            # Calculate Euler angles (yaw, pitch, roll) from the rotation vector
            yaw, pitch, roll = rotation_vector_to_euler_angles(rotation_vector)

            # Write the results to the CSV file
            with open(str(Path(HEADPOSE_PATH, "headpose_values.csv")), mode='a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([participant_id, yaw[0], pitch[0], roll[0]])