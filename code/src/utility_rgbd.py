import cv2
import numpy as np
import dlib
import csv 


def rotation_matrix_to_euler_angles(matrix):
    """Convert a rotation matrix to Euler angles (roll, pitch, yaw)"""
    sy = np.sqrt(matrix[0,0] * matrix[0,0] +  matrix[1,0] * matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2,1] , matrix[2,2])
        y = np.arctan2(-matrix[2,0], sy)
        z = np.arctan2(matrix[1,0], matrix[0,0])
    else:
        x = np.arctan2(-matrix[1,2], matrix[1,1])
        y = np.arctan2(-matrix[2,0], sy)
        z = 0
    return np.array([x, y, z])


# Create a Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/cko4/2023_asd_gaze/code/src/shape_predictor_68_face_landmarks.dat')


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


def get_image_points_and_model_points(color_image, face, depth_image):
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
    depth_image[landmarks.part(30).x, landmarks.part(30).y],
    depth_image[landmarks.part(36).x, landmarks.part(36).y],
    depth_image[landmarks.part(45).x, landmarks.part(45).y],
    depth_image[(landmarks.part(48).x + landmarks.part(54).x) // 2, (landmarks.part(48).y + landmarks.part(54).y) // 2],
    depth_image[landmarks.part(48).x, landmarks.part(48).y],
    depth_image[landmarks.part(54).x, landmarks.part(54).y],
]

    print("Depth Values: ", depths)

    # Create 3D model points using depth information
    model_points = model_points_without_depth.copy()
    model_points[:, 2] += depths

    print("Image Points: ", image_points)
    print("Model Points: ", model_points)

    return image_points, model_points


def draw_face_bounding_boxes(color_image, faces):
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return color_image


def write_headpose_to_csv(csv_path, participant_id, roll, pitch, yaw):
    with open(csv_path, mode='a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([participant_id, roll, pitch, yaw])
