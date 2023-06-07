#%%
import cv2
import os 
import numpy as np
from pathlib import Path
from src.utility import rotation_vector_to_euler_angles, camera_matrix, detector, get_image_points_and_model_points, draw_face_bounding_boxes, write_headpose_to_csv 

ROOT_PATH = Path("/mnt/2021_NIA_data/projects/nbb")
VIDEO_PATH = ROOT_PATH.joinpath("video")
PROC_DATA_PATH = VIDEO_PATH.joinpath("proc_data")
PROJECT_PATH = Path(__file__).parents[1]
IMAGE_PATH = Path(PROJECT_PATH, "images")
PROC_PARTICIPANT_PATH = Path(PROJECT_PATH, "data")
HEADPOSE_PATH = Path(PROJECT_PATH, "data/headpose_data")

# Load previously processed participants if any
processed_participants_file = PROC_PARTICIPANT_PATH.joinpath("processed_participants.txt")
processed_participants = set()
if processed_participants_file.exists():
    with open(processed_participants_file, 'r') as f:
        processed_participants = set(line.strip() for line in f)

headpose_csv_file = Path(HEADPOSE_PATH, "headpose_values.csv")

# Check if the file exists. If not, write the header.
if not os.path.exists(headpose_csv_file):
    with open(headpose_csv_file, 'w') as f:
        f.write('participant_id,yaw,pitch,roll\n')

#%%
# Iterate through each depth and RGB file
for color_file in Path(PROC_DATA_PATH, "color").glob("*.npy"):
    # Get the participant ID from the file name
    participant_id = color_file.stem.split("_")[0]
    
    # Check if participant already processed
    if participant_id in processed_participants:
        continue

    depth_file = Path(PROC_DATA_PATH, "depth", color_file.stem.replace("color", "depth") + ".npy")

    if depth_file.exists():
        depth_images = np.load(depth_file)
        color_images = np.load(color_file)

        # Iterate through each frame
        for frame_idx, (depth_image, color_image) in enumerate(zip(depth_images, color_images)):
            # print(f"Processing frame {frame_idx}")
            
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the image
            faces = detector(color_image, 1)
            # print(f"Frame {frame_idx}: Number of detected faces:", len(faces))
            color_image = draw_face_bounding_boxes(color_image, faces)

            # Save every 50th frame as an image file
            # if frame_idx % 50 == 0:  # Save every 50th frame
            #     cv2.imwrite(str(Path(IMAGE_PATH,f"{participant_id}_frame_{frame_idx}.png")), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            for face in faces:
                image_points, model_points = get_image_points_and_model_points(color_image, face, depth_image)
                _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)
                pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)

                write_headpose_to_csv(str(Path(HEADPOSE_PATH, "headpose_values.csv")), participant_id, roll, pitch, yaw)
        
        # After all frames are processed, add participant_id to processed_participants and write to file
        processed_participants.add(participant_id)
        with open(processed_participants_file, 'a') as f:
            f.write(f"{participant_id}\n")


# %%
# import cv2
# import numpy as np
# from pathlib import Path
# from src.utility import rotation_vector_to_euler_angles, camera_matrix, detector, get_image_points_and_model_points, draw_face_bounding_boxes, write_headpose_to_csv 

# ROOT_PATH = Path('/mnt/2021_NIA_data/projects/nbb')
# VIDEO_PATH = ROOT_PATH.joinpath("video")
# PROC_DATA_PATH = VIDEO_PATH.joinpath("proc_data")
# HEADPOSE_PATH = VIDEO_PATH.joinpath("headpose_data")
# IMAGE_PATH = VIDEO_PATH.joinpath("images")

# # Load the color images
# color_file = Path(PROC_DATA_PATH, "color", "td006_color.npy")
# depth_file = Path(PROC_DATA_PATH, "depth", color_file.stem.replace("color", "depth") + ".npy")
# color_images = np.load(color_file)
# participant_id = color_file.stem.split("_")[0]

# if depth_file.exists():
#     depth_images = np.load(depth_file)
#     color_images = np.load(color_file)
#     # Iterate through each image
#     for frame_idx, (depth_image, color_image) in enumerate(zip(depth_images, color_images)):  # Corrected this line
#         # Convert the color image to RGB
#         color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
#         # Detect faces in the image
#         faces = detector(color_image, 1)
        
#         for face in faces:
#             image_points, model_points = get_image_points_and_model_points(color_image, face, depth_image)
#             _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)
#             yaw, pitch, roll = rotation_vector_to_euler_angles(rotation_vector)

#             write_headpose_to_csv(str(Path(HEADPOSE_PATH, "headpose_values.csv")), participant_id, yaw, pitch, roll)