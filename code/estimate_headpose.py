#%%
import cv2
import numpy as np
from pathlib import Path
from src.utility import rotation_vector_to_euler_angles, camera_matrix, detector, get_image_points_and_model_points, draw_face_bounding_boxes, write_headpose_to_csv 

PROJECT_PATH = Path(__file__).parents[1]
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")
HEADPOSE_PATH = Path(PROJECT_PATH, "data/headpose_data")
IMAGE_PATH = Path(PROJECT_PATH, "data/images")

#%%
# Iterate through each depth and RGB file
for color_file in Path(PROC_DATA_PATH, "rgb").glob("*.npy"):
    depth_file = Path(PROC_DATA_PATH, "depth", color_file.stem.replace("color", "depth") + ".npy")

    if depth_file.exists():
        depth_images = np.load(depth_file)
        color_images = np.load(color_file)

        # Get the participant ID from the file name
        participant_id = color_file.stem.split("_")[0]

        # unique_color_frames, unique_indices_color = np.unique(color_images, return_index=True, axis=0)
        # unique_depth_frames, unique_indices_depth = np.unique(depth_images, return_index=True, axis=0)
        # print(f"{participant_id}: Unique color frames: {len(unique_color_frames)}, Unique depth frames: {len(unique_depth_frames)}")

        # Iterate through each frame
        for frame_idx, (depth_image, color_image) in enumerate(zip(depth_images, color_images)):
            # print(f"Processing frame {frame_idx}")
            
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the image
            faces = detector(color_image, 1)
            # print(f"Frame {frame_idx}: Number of detected faces:", len(faces))
            color_image = draw_face_bounding_boxes(color_image, faces)

            # Save every 50th frame as an image file
            if frame_idx % 50 == 0:  # Save every 50th frame
                cv2.imwrite(str(Path(IMAGE_PATH,f"frame_{participant_id}_{frame_idx}.png")), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            for face in faces:
                image_points, model_points = get_image_points_and_model_points(color_image, face, depth_image)
                _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)
                yaw, pitch, roll = rotation_vector_to_euler_angles(rotation_vector)

                write_headpose_to_csv(str(Path(HEADPOSE_PATH, "headpose_values.csv")), participant_id, yaw, pitch, roll)


# import matplotlib.pyplot as plt 
# color_file1 = Path(PROC_DATA_PATH, "rgb/td001_gaze_color.npy")
# depth_file1 = Path(PROC_DATA_PATH, "depth/td001_gaze_depth.npy")
# depth_images1 = np.load(depth_file1)
# color_images1 = np.load(color_file1)
# # Display the first frame of both depth and color images
# plt.subplot(1, 2, 1)
# plt.imshow(depth_images1[30], cmap="gray")
# plt.title("Depth Frame 30")
# plt.subplot(1, 2, 2)
# plt.imshow(color_images1[30])
# plt.title("Color Frame 30")