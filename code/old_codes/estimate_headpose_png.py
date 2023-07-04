import cv2
import numpy as np
from pathlib import Path
from code.src.utility_sample import rotation_vector_to_euler_angles, camera_matrix, model_points_without_depth, detector, predictor 

PROJECT_PATH = Path(__file__).parents[1]
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")

# Load the depth and color images
depth_image = cv2.imread(str(Path(PROC_DATA_PATH, 'sample_depth1.png')), cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread(str(Path(PROC_DATA_PATH, 'sample_color1.png'))) 

# Detect faces in the color image
faces = detector(color_image)

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

    # Print the results
    print("Yaw: {:.2f} degrees, Pitch: {:.2f} degrees, Roll: {:.2f} degrees".format(yaw[0], pitch[0], roll[0]))
