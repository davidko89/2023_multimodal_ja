import cv2
import numpy as np 
import os
from pathlib import Path
from src.utility_rgb import rotation_matrix_to_euler_angles, camera_matrix, detector, get_image_points_and_model_points, draw_face_bounding_boxes, write_headpose_to_csv

ROOT_PATH = Path("/mnt/2021_NIA_data/projects/nbb")
RAW_DATA_PATH = ROOT_PATH.joinpath("video/raw_data")
PROC_DATA_PATH = ROOT_PATH.joinpath("video/proc_data")
PROJECT_PATH = Path(__file__).parents[1]
IMAGE_PATH = Path(PROJECT_PATH, "images")
PROC_PARTICIPANT_PATH = Path(PROJECT_PATH, "data")
HEADPOSE_PATH = Path(PROJECT_PATH, "data/headpose_data")
headpose_csv_file = Path(HEADPOSE_PATH, "headpose_values_rgb.csv")


def check_if_processed(participant_id):
    processed_participants_file = PROC_PARTICIPANT_PATH.joinpath("processed_participants_rgb.txt")
    processed_participants = set()
    if processed_participants_file.exists():
        with open(processed_participants_file, 'r') as f:
            processed_participants = set(line.strip() for line in f)
    
    return processed_participants
           

def get_video_path(participant_id):
    """Get the path to the video for each participant"""
    video_path = RAW_DATA_PATH.joinpath(participant_id, f"{participant_id}.mp4")
    
    return video_path


def read_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

    fc = 0
    ret = True

    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    
    video_array = np.transpose(buf, (0, 3, 2, 1))
    
    return video_array


def estimate_headpose_rgb(video_path, participant_id):
    pass 
    



def main():
    for video_file in Path(RAW_DATA_PATH).glob("*.mp4"):
        participant_id = video_file.stem[0]

        processed_participants = check_if_processed(participant_id)
        if participant_id in processed_participants:
            continue

        video_path = get_video_path(participant_id)

        roll, pitch, yaw = estimate_headpose_rgb(video_path, participant_id)

        write_headpose_to_csv(str(headpose_csv_file), participant_id, roll, pitch, yaw) 
    

if __name__ == "__main__":
    main()