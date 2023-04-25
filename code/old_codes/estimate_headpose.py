#%%
import cv2
import dlib
import numpy as np
from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]
RAW_DATA_PATH = Path(PROJECT_PATH, "data/raw_data")
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")
FACE_LANDMARKS_PATH = Path(PROJECT_PATH, "code/src/shape_predictor_68_face_landmarks.dat")

#%%
# Load the depth and color images
depth_image = cv2.imread(str(Path(PROC_DATA_PATH, 'sample_depth1.png')), cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread(str(Path(PROC_DATA_PATH, 'sample_color1.png'))) 

# Create a Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(Path(FACE_LANDMARKS_PATH)))

# Detect faces in the color image
faces = detector(color_image)

# Iterate through each detected face
for face in faces:
    # Draw a rectangle around the face
    cv2.rectangle(color_image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

    # Get the landmarks for the face
    landmarks = predictor(color_image, face)
    
    # Extract the relevant landmarks for head pose estimation
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

    # Draw circles at the landmark points
    for point in [nose_tip, left_eye, right_eye]:
        cv2.circle(color_image, tuple(point), 3, (0, 255, 0), -1)
    
    # Calculate the 3D position of the landmarks using the depth image
    nose_tip_depth = depth_image[nose_tip[1], nose_tip[0]]
    left_eye_depth = depth_image[left_eye[1], left_eye[0]]
    right_eye_depth = depth_image[right_eye[1], right_eye[0]]
    
    # Calculate the head pose angles using the 3D positions of the landmarks
    # (note that these equations assume a pinhole camera model)
    fx = fy = 500  # focal length in pixels
    cx = color_image.shape[1] / 2  # image center x-coordinate
    cy = color_image.shape[0] / 2  # image center y-coordinate
    
    yaw = np.arctan2(nose_tip[0] - cx, fx) - np.arctan2(right_eye[0] - left_eye[0], fx)
    pitch = np.arctan2(nose_tip_depth - (left_eye_depth + right_eye_depth) / 2, fy)
    
    # Print the results
    print("Yaw: {:.2f} degrees, Pitch: {:.2f} degrees".format(yaw * 180 / np.pi, pitch * 180 / np.pi))

# Show the resulting image with the detected faces and landmarks
cv2.imwrite(str(Path(PROC_DATA_PATH, 'output_color_image.png')), color_image)
