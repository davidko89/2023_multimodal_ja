import cv2
import numpy as np
import dlib
import csv
import os  

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
    return np.array([x, y, z]) * (180. / np.pi)  # Convert to degrees


# Create a Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/cko4/2023_asd_gaze/code/src/shape_predictor_68_face_landmarks.dat')


# Camera intrinsic parameters (Sony Alpha a5000)
fx, fy = 2454.34, 2075.48  # focal length in pixels
cx, cy = 959.16, 539.63 # Placeholder values for image center (cx, cy)

# Function to update the camera matrix based on the input image shape
def update_camera_matrix(image_shape):
    global cx, cy, camera_matrix
    cx, cy = image_shape[1] / 2, image_shape[0] / 2  # image center
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return camera_matrix


# Define 3D model points
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ], dtype=np.float32)


def get_image_points_and_model_points(video_image, face):
    # Get the landmarks for the face
    landmarks = predictor(video_image, face)

    # Extract the relevant landmarks for head pose estimation (2D image points)
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),  # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
        (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
    ], dtype=np.float32)

    return image_points, model_points


def draw_face_bounding_boxes(video_image, faces):
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(video_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return video_image


def write_headpose_to_csv(csv_path, participant_id, frame_idx, roll, pitch, yaw):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a') as csv_file:
        headers = ['participant_id', 'frame_idx', 'roll', 'pitch', 'yaw']
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'participant_id': participant_id,
            'frame_idx': frame_idx,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
        })
