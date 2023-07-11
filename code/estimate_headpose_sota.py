import cv2
import numpy as np 
from pathlib import Path
from src.utility_rgb_test2 import detector, draw_face_bounding_boxes, write_headpose_to_csv
from sixdrepnet import SixDRepNet
from tqdm import tqdm

ROOT_PATH = Path("/mnt/2021_NIA_data/projects/nbb")
RAW_DATA_PATH = ROOT_PATH.joinpath("video/raw_data")
PROC_DATA_PATH = ROOT_PATH.joinpath("video/proc_data")
PROJECT_PATH = Path(__file__).parents[1]
IMAGE_PATH = Path(PROJECT_PATH, "images_model_sota")
PROC_PARTICIPANT_PATH = Path(PROJECT_PATH, "data")


def check_if_processed(participant_id):
    processed_participants_file = PROC_PARTICIPANT_PATH.joinpath("processed_participants_rgb_sota.txt")
    if processed_participants_file.exists():
        with open(processed_participants_file, 'r') as f:
            processed_participants = set(line.strip() for line in f)
            return participant_id in processed_participants
    return False
           

def mark_participant_as_processed(participant_id):
    processed_participants_file = PROC_PARTICIPANT_PATH.joinpath("processed_participants_rgb_sota.txt")
    with open(processed_participants_file, 'a') as f:
        f.write(f"{participant_id}\n")


def get_video_path(participant_id):
    """Get the path to the video for each participant"""
    video_path = RAW_DATA_PATH.joinpath(participant_id, f"{participant_id}.mp4")
    return video_path


def read_video(video_file: Path):
    cap = cv2.VideoCapture(str(video_file))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
     
    buf = []
    
    for fc in range(frameCount):
        ret, frame = cap.read()
        if fc % 10 == 0:
            buf.append(frame)

    cap.release()

    video_array = np.array(buf, dtype="uint8")
    return video_array


def estimate_headpose_rgb(video_file, participant_id):
    headpose_results = []  # Store head pose results for each face
    model = SixDRepNet() # Initialize SixDRepNet model

    cap = cv2.VideoCapture(str(video_file))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in tqdm(range(frameCount), desc="Processing frames"):
        ret, frame = cap.read()

        # Skip frames that aren't read correctly
        if not ret:
            continue

        # Only process every 10th frame
        if frame_idx % 10 != 0:
            continue

        video_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector(video_image, 1)
        video_image = draw_face_bounding_boxes(video_image, faces)

        # Skip frame if no faces are detected
        if not faces:
            continue

        # Process each detected face
        for face in faces:
            try:
                # Crop the face from the image
                face_crop = video_image[face.top():face.bottom(), face.left():face.right()]

                # Estimate pitch, yaw, and roll using SixDRepNet model
                pitch, yaw, roll = model.predict(face_crop)
                
                # Calculate the center of the face
                tdx = int(face.left() + (face.right() - face.left()) / 2)
                tdy = int(face.top() + (face.bottom() - face.top()) / 2)

                model.draw_axis(video_image, yaw, pitch, roll, tdx, tdy)

                # Add head pose results to the list
                headpose_results.append((frame_idx, roll, pitch, yaw))

                # Save every 100th frame as image files
                if frame_idx % 100 == 0 or frame_idx == 0:  
                    cv2.imwrite(str(Path(IMAGE_PATH, f"{participant_id}_frame_{frame_idx}.png")), cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error while processing face on frame {frame_idx}: {e}")

    cap.release()

    return headpose_results


def main():
    for video_file in Path(RAW_DATA_PATH).glob("*.mp4"):
        print(f"Processing file: {video_file}")
        participant_id = video_file.stem

        if check_if_processed(participant_id):
            print(f"Skipping processed participant: {participant_id}")
            continue

        try:
            headpose_results = estimate_headpose_rgb(video_file, participant_id)
        
        except Exception as e:
            print(f"Error during head pose estimation for participant {participant_id}: {e}")
            continue
        
        # Write head pose results to csv file
        for frame_idx, roll, pitch, yaw in headpose_results:
            print("Writing to CSV...")
            headpose_dir = PROC_PARTICIPANT_PATH.joinpath("headpose_data")
            headpose_dir.mkdir(parents=True, exist_ok=True)
            headpose_csv_file = headpose_dir.joinpath("headpose_values_rgb_sota_fixed.csv") # May need to change this 
            write_headpose_to_csv(headpose_csv_file, participant_id, frame_idx, roll, pitch, yaw)

        # Mark participant as processed
        mark_participant_as_processed(participant_id)
        print(f"Finished processing for participant_id: {participant_id}")


if __name__ == "__main__":
    main()