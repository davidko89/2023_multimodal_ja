import cv2
import numpy as np
import dlib
import csv 


def rotation_vector_to_euler_angles(rotation_vector):
    """Convert a rotation vector to Euler angles (yaw, pitch, and roll)."""
    matrix, _ = cv2.Rodrigues(rotation_vector)
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((matrix, np.zeros((3, 1)))))
    return euler_angles


# Create a Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/cko4/2023_asd_gaze/code/src/shape_predictor_68_face_landmarks.dat')


# Camera intrinsic parameters (you should use the actual values for your camera)
fx, fy = 961.8638, 550.6279  # focal length in pixels
cx, cy = 912.718, 912.5225  # image center
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# fx, fy = 500, 500  # focal length in pixels

# # Placeholder values for image center (cx, cy)
# cx, cy = 0, 0

# # Function to update the camera matrix based on the input image shape
# def update_camera_matrix(image_shape):
#     global cx, cy, camera_matrix
#     cx, cy = image_shape[1] / 2, image_shape[0] / 2  # image center
#     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
#     return camera_matrix

# # Initialize camera_matrix with default values
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


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

    return image_points, model_points


def draw_face_bounding_boxes(color_image, faces):
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return color_image


def write_headpose_to_csv(csv_path, participant_id, yaw, pitch, roll):
    with open(csv_path, mode='a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([participant_id, yaw[0], pitch[0], roll[0]])
