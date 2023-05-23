#%%
import open3d as o3d
import numpy as np
from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]
RAW_DATA_PATH = Path(PROJECT_PATH, "data/raw_data")
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")

#%%
reader = o3d.io.AzureKinectMKVReader()
reader.open(str(Path(RAW_DATA_PATH, "sample_output.mkv")))

color_image = []
depth_image = []
flag = 0
while(True):
    if reader.is_eof():
        break

    frame = reader.next_frame()
    
    if frame == None:
        continue
    
    # if flag == 70:
    #     o3d.io.write_image(f'{PROC_DATA_PATH}/sample_color.png', frame.color)
    #     o3d.io.write_image(f'{PROC_DATA_PATH}/sample_depth.png', frame.depth)
    #     flag = 10000

    color_image.append(np.asarray(frame.color))
    depth_image.append(np.asarray(frame.depth))
    # flag += 1


color_image = np.array(color_image)
depth_image = np.array(depth_image)
print(np.max(depth_image[0]))
print(f'Color shape: {color_image.shape}\nDepth shape: {depth_image.shape}')
  

#%%
# for i in range(len(target_frames)):
#     # Extract the color and depth images from the frame
#     color_image = np.asarray(target_frames[i].color)
#     depth_image = np.asarray(target_frames[i].depth)

#     # Save the images to the proc_data folder
#     color_path = Path(PROC_DATA_PATH, f"color_{i}.png")
#     depth_path = Path(PROC_DATA_PATH, f"depth_{i}.png")
#     o3d.io.write_image(str(color_path), color_image)
#     o3d.io.write_image(str(depth_path), depth_image)

# %%
