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
FACE_LANDMARKS_PATH = Path(PROJECT_PATH, "code/src/shape_predictor_68_face_landmarks.dat")

# Load the depth and color images
depth_image = cv2.imread(str(Path(PROC_DATA_PATH, 'sample_depth1.png')), cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread(str(Path(PROC_DATA_PATH, 'sample_color1.png'))) 

# Create a Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(Path(FACE_LANDMARKS_PATH)))

# Detect faces in the color image
faces = detector(color_image)

# Camera intrinsic parameters (you should use the actual values for your camera)
fx, fy = 500, 500  # focal length in pixels
cx, cy = color_image.shape[1] / 2, color_image.shape[0] / 2  # image center
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# Define 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),    # Nose tip
    (-30.0, -20.0, -30.0),   # Left eye
    (30.0, -20.0, -30.0),    # Right eye
    (0.0, 50.0, -50.0),   # Mouth center
    (-20.0, 50.0, -50.0),   # Left mouth corner
    (20.0, 50.0, -50.0),    # Right mouth corner
], dtype=np.float32)

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


    # Estimate head pose using solvePnP
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)

    # Calculate Euler angles (yaw, pitch, roll) from the rotation vector
    yaw, pitch, roll = rotation_vector_to_euler_angles(rotation_vector)

    # Print the results
    print("Yaw: {:.2f} degrees, Pitch: {:.2f} degrees, Roll: {:.2f} degrees".format(yaw[0], pitch[0], roll[0]))
