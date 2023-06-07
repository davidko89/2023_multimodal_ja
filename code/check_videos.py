#%%
import cv2
import numpy as np
from pathlib import Path
from src.utility import detector, draw_face_bounding_boxes

ROOT_PATH = Path("/mnt/2021_NIA_data/projects/nbb")
VIDEO_PATH = ROOT_PATH.joinpath("video")
PROC_DATA_PATH = VIDEO_PATH.joinpath("proc_data")
PROJECT_PATH = Path(__file__).parents[1]
IMAGE_PATH = Path(PROJECT_PATH, "images")

#%%
# Load the color images
color_file = Path(PROC_DATA_PATH, "color", "td009_color.npy")
depth_file = Path(PROC_DATA_PATH, "depth", color_file.stem.replace("color", "depth") + ".npy")
color_images = np.load(color_file)
participant_id = color_file.stem.split("_")[0]

if depth_file.exists():
    depth_images = np.load(depth_file)
    color_images = np.load(color_file)
    
    # Get the total number of frames
    total_frames = len(color_images)
    print("Total frames:", total_frames)
    
    # Iterate through each image
    for frame_idx, (depth_image, color_image) in enumerate(zip(depth_images, color_images)):
        # print(f"Processing frame {frame_idx+1}/{total_frames}")
        
        # Convert the color image to RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        faces = detector(color_image, 1)
        # print(f"Frame {frame_idx}: Number of detected faces:", len(faces))
        
        color_image = draw_face_bounding_boxes(color_image, faces)

        # Save every 50th frame as an image file
        if frame_idx % 50 == 0:  # Save every 50th frame
            cv2.imwrite(str(Path(IMAGE_PATH,f"{participant_id}_frame_{frame_idx}.png")), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

#%%
# Iterate through each depth and RGB file
for color_file in Path(PROC_DATA_PATH, "color").glob("*.npy"):
    depth_file = Path(PROC_DATA_PATH, "depth", color_file.stem.replace("color", "depth") + ".npy")

    if depth_file.exists():
        depth_images = np.load(depth_file)
        color_images = np.load(color_file)

        # Get the participant ID from the file name
        participant_id = color_file.stem.split("_")[0]

        # Iterate through each frame
        for frame_idx, (depth_image, color_image) in enumerate(zip(depth_images, color_images)):
            # print(f"Processing frame {frame_idx}")
            
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the image
            faces = detector(color_image, 1)
            # print(f"Frame {frame_idx}: Number of detected faces:", len(faces))
            color_image = draw_face_bounding_boxes(color_image, faces)

            # Save every 10th frame as an image file
            if frame_idx % 10 == 0:  # Save every 10th frame
                cv2.imwrite(str(Path(IMAGE_PATH,f"{participant_id}_frame_{frame_idx}.png")), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
# %%
