import numpy as np
import pandas as pd
import os
import cv2
import glob

from collections import defaultdict
from typing import List, Tuple, Optional

SIZE = 20
VARIANCE = 10
WIDTH = 1280
HEIGHT = 720   


def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def yolo_to_pixel_coords(x_center: float, y_center: float) -> Tuple[int, int]:
    """
    Convert YOLO format normalized coordinates (0-1) to pixel coordinates.
    """
    pixel_x = int(x_center * WIDTH)
    pixel_y = int(y_center * HEIGHT)
    return pixel_x, pixel_y

def read_yolo_annotation(file_path: str) -> Optional[Tuple[int, int]]:
    """
    Read YOLO format annotation file and return the center coordinates in pixels.
    Returns None if the file is empty (no ball visible).
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if content:  # Check if file is not empty
                # Parse YOLO format: class x_center y_center width height
                parts = content.split()
                if len(parts) == 5 and parts[0] == '0':  # Class 0 for ball
                    # Convert normalized coordinates to pixel coordinates
                    x_norm, y_norm = float(parts[1]), float(parts[2])
                    return yolo_to_pixel_coords(x_norm, y_norm)
    except Exception as e:
        print(f"Error reading annotation file {file_path}: {e}")
    return None

def create_gt_image(x: int, y: int, width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Create a black and white image with a single white pixel at (x,y)
    """
    # Create black image
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    gaussian_kernel_array = create_gaussian(SIZE, VARIANCE)
    for i in range(-SIZE, SIZE+1):
        for j in range(-SIZE, SIZE+1):
                if x+i<width and x+i>=0 and y+j<height and y+j>=0 :
                    temp = gaussian_kernel_array[i+SIZE][j+SIZE]
                    if temp > 0:
                        heatmap[y+j,x+i] = (temp,temp,temp)
    return heatmap

def get_frame_groups(clip_path: str, sequence_length: int = 3) -> List[Tuple[List[str], str, str]]:
    """
    Create groups of consecutive frames and their corresponding label files.
    Returns list of tuples: (frame_paths, last_frame_path, last_label_path)
    """
    # Get sorted lists of image and label files
    images_dir = clip_path
    labels_dir = '/Users/cosmincojocaru/playground/ball/nba_ball_dataset/labels/train'
    
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    
    frame_groups = []
    for i in range(len(image_files) - sequence_length + 1):
        group_frames = image_files[i:i + sequence_length]
        last_frame = group_frames[-1]
        # Convert last frame image path to corresponding label path
        last_label = os.path.join(labels_dir, last_frame.split('/')[-1].replace('.jpg', '.txt'))
        
        if os.path.exists(last_label):
            frame_groups.append((group_frames, last_frame, last_label))
    
    return frame_groups

def create_gt_labels(path_input, path_output, train_rate=0.7):
    df = pd.DataFrame()
    dataset_rows = []
    
    # Process each clip directory
    clip_dirs = sorted(glob.glob(os.path.join(path_input, 'subclip*')))
    
    for clip_dir in clip_dirs:
        clip_name = os.path.basename(clip_dir)
        print(f"Processing {clip_name}...")
        
        frame_groups = get_frame_groups(clip_dir)
        for group_frames, last_frame, last_label in frame_groups:
            # Get ball coordinates from last frame's label (now in pixels)
            pixel_coords = read_yolo_annotation(last_label)
            
            if pixel_coords is not None:  # Only include groups where ball is visible in last frame
                x_pixel, y_pixel = pixel_coords
                
                # Create row with relative paths to frames and pixel coordinates
                row = {
                    'clip_name': clip_name,
                    'frame1': os.path.basename(group_frames[0]),
                    'frame2': os.path.basename(group_frames[1]),
                    'frame3': os.path.basename(group_frames[2]),
                    'ball_x': x_pixel,
                    'ball_y': y_pixel
                }
                dataset_rows.append(row)
    # Create DataFrame and save to CSV
    df = pd.DataFrame(dataset_rows)
    out_csv = os.path.join(path_output, 'labels.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nDataset created successfully: {path_output}")
    print(f"Total sequences: {len(df)}")

def process_dataset(csv_path: str, base_dir: str):
    """
    Process the dataset to create ground truth images and update the CSV.
    """
    # Read the existing CSV
    df = pd.read_csv(csv_path)
    
    # Create a directory for ground truth images if it doesn't exist
    for clip_name in df['clip_name'].unique():
        gt_dir = os.path.join(base_dir, clip_name, 'gts')
        os.makedirs(gt_dir, exist_ok=True)
    
    # Create a mapping of frame to coordinates for each clip
    frame_coords = defaultdict(dict)
    
    # Process each row in the dataset to build the mapping
    for _, row in df.iterrows():
        clip_name = row['clip_name']
        # Add coordinates for frame3 (last frame in sequence)
        frame_coords[clip_name][row['frame3']] = (row['ball_x'], row['ball_y'])
        
        # For frame2 (middle frame), take coordinates if it's the last frame of another sequence
        if _ > 0:  # Not the first row
            prev_row = df.iloc[_ - 1]
            if prev_row['clip_name'] == clip_name and prev_row['frame3'] == row['frame2']:
                frame_coords[clip_name][row['frame2']] = (prev_row['ball_x'], prev_row['ball_y'])
        
        # For frame1 (first frame), take coordinates if it's the last frame of another sequence
        if _ > 1:  # Not the first or second row
            prev_prev_row = df.iloc[_ - 2]
            if prev_prev_row['clip_name'] == clip_name and prev_prev_row['frame3'] == row['frame1']:
                frame_coords[clip_name][row['frame1']] = (prev_prev_row['ball_x'], prev_prev_row['ball_y'])

    # Create ground truth images for all frames that have coordinates
    for clip_name, frames in frame_coords.items():
        print(f"Processing ground truth images for {clip_name}...")
        gt_dir = os.path.join(base_dir, clip_name, 'gts')
        
        for frame_name, (x, y) in frames.items():
            # Create the ground truth image
            gt_img = create_gt_image(int(x), int(y))
            
            # Save the ground truth image with same name as original frame but .png extension
            gt_filename = frame_name.replace('.jpg', '.png')
            gt_path = os.path.join(gt_dir, gt_filename)
            cv2.imwrite(gt_path, gt_img)
    
    # Add ground truth paths to the DataFrame
    df['gt_path'] = df.apply(
        lambda row: os.path.join(
            row['clip_name'], 
            'gts', 
            row['frame3'].replace('.jpg', '.png')
        ),
        axis=1
    )
    
    # Save the updated CSV
    output_csv = csv_path.replace('.csv', '_with_gt.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nUpdated dataset saved to: {output_csv}")
    
    # Print statistics
    total_gt_images = sum(len(frames) for frames in frame_coords.values())
    print(f"\nStatistics:")
    print(f"Total ground truth images created: {total_gt_images}")
    print(f"Number of clips processed: {len(frame_coords)}")
    
    # Verify all ground truth images exist
    missing_gts = []
    for _, row in df.iterrows():
        gt_path = os.path.join(base_dir, row['gt_path'])
        if not os.path.exists(gt_path):
            missing_gts.append(gt_path)
    
    if missing_gts:
        print("\nWarning: Some ground truth images are missing:")
        for path in missing_gts[:5]:  # Show first 5 missing files
            print(f"- {path}")
        if len(missing_gts) > 5:
            print(f"... and {len(missing_gts) - 5} more")

if __name__ == '__main__':
    path_input = './nba_dataset_720'
    path_output = './nba_dataset/gts'
    
    # if not os.path.exists(path_output):
    #     os.makedirs(path_output)
        
    create_gt_labels(path_input, path_output)

    csv_file = './nba_dataset_720/gts/labels.csv'
    process_dataset(csv_file, path_input)

                            