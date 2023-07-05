import cv2
import numpy as np 
from pathlib import Path
from src.utility_rgb import rotation_matrix_to_euler_angles, update_camera_matrix, detector, get_image_points_and_model_points, draw_face_bounding_boxes, write_headpose_to_csv

ROOT_PATH = Path("/mnt/2021_NIA_data/projects/nbb")
RAW_DATA_PATH = ROOT_PATH.joinpath("video/raw_data")
PROC_DATA_PATH = ROOT_PATH.joinpath("video/proc_data")
PROJECT_PATH = Path(__file__).parents[1]
IMAGE_PATH = Path(PROJECT_PATH, "images")
PROC_PARTICIPANT_PATH = Path(PROJECT_PATH, "data")


def check_if_processed(participant_id):
    processed_participants_file = PROC_PARTICIPANT_PATH.joinpath("processed_participants_rgb.txt")
    if processed_participants_file.exists():
        with open(processed_participants_file, 'r') as f:
            processed_participants = set(line.strip() for line in f)
            return participant_id in processed_participants
    return False
           

def mark_participant_as_processed(participant_id):
    processed_participants_file = PROC_PARTICIPANT_PATH.joinpath("processed_participants_rgb.txt")
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
        if fc % 3 == 0:
            buf.append(frame)

    cap.release()

    video_array = np.array(buf, dtype="uint8")

    return video_array


def estimate_headpose_rgb(video_file, participant_id):
    headpose_results = []  # Store head pose results for each face

    for frame_idx, video_image in enumerate(read_video(video_file)):
        video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        faces = detector(video_image, 1)
        video_image = draw_face_bounding_boxes(video_image, faces)

        # Save every 100th frame as image files
        if frame_idx % 100 == 0 or frame_idx == 0:
            cv2.imwrite(str(Path(IMAGE_PATH, f"{participant_id}_frame_{frame_idx}.png")), cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR))

        # Skip frame if no faces are detected
        if not faces:
            continue

        # Process each detected face
        for face in faces:
            try:
                image_points, model_points = get_image_points_and_model_points(video_image, face)

                # Solve for pose
                camera_matrix = update_camera_matrix(video_image.shape)
                _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, None)

                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Convert rotation matrix to euler angles
                roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

                # Add head pose results to the list
                headpose_results.append((frame_idx, roll, pitch, yaw))
                # print(headpose_results)
                # print(f"Head pose results added for frame {frame_idx}")

            except Exception as e:
                print(f"Error while processing face on frame {frame_idx}: {e}")

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
            headpose_csv_file = headpose_dir.joinpath("headpose_values_rgb.csv")
            write_headpose_to_csv(headpose_csv_file, participant_id, frame_idx, roll, pitch, yaw)

        # Mark participant as processed
        mark_participant_as_processed(participant_id)

        print(f"Finished processing for participant_id: {participant_id}")


if __name__ == "__main__":
    main()


# def read_video(video_path: Path):
#     print(f"Reading video from {video_path}")
#     if not video_path.exists():
#         print(f"Video file not found at {video_path}")
#         return None

#     cap = cv2.VideoCapture(str(video_path))

#     if not cap.isOpened():
#         print("Failed to open video")
#         return None

#     frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Frame count: {frameCount}, Frame width: {frameWidth}, Frame height: {frameHeight}")

#     buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

#     fc = 0
#     ret = True

#     while fc < frameCount and ret:
#         ret, buf[fc] = cap.read()
#         if not ret:
#             print(f"Failed to read frame {fc}")
#         fc += 1

#     cap.release()

#     video_array = np.transpose(buf, (0, 3, 2, 1))
#     print(f"Video array shape: {video_array.shape}, dtype: {video_array.dtype}")

#     return video_array