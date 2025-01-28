import os
import cv2
import numpy as np
from pathlib import Path

def create_ball_trajectory_video(frames_dir, labels_dir, output_path, fps=30):
    """
    Create a video showing the ball trajectory for a given subclip.
    
    Args:
        frames_dir (str): Path to directory containing frame images
        labels_dir (str): Path to directory containing YOLO annotations
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
    """
    # Get all frame files and sort them
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_file in frame_files:
        # Read frame
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        
        # Get corresponding label file
        label_file = frame_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            # Read YOLO format annotation
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:  # Check if file is not empty
                    # Parse YOLO format (class x_center y_center width height)
                    _, x_center, y_center, _, _ = map(float, line.split())
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = int(x_center * width)
                    y = int(y_center * height)
                    
                    # Draw circle at ball position
                    cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()

def process_all_subclips(dataset_root, output_root):
    """
    Process all subclips in the dataset.
    
    Args:
        dataset_root (str): Root directory containing liga_1_ball_dataset and liga_1_ball_labels
        output_root (str): Directory where output videos will be saved
    """
    frames_root = os.path.join(dataset_root, 'liga_1_dataset_1')
    labels_root = os.path.join(dataset_root, '/Users/cosmincojocaru/playground/ball/liga_1_ball_dataset/labels/train')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    subclip = 'subclip_4125'
    frames_dir = os.path.join(frames_root, subclip)
    for subclip in os.listdir(frames_root):
        frames_dir = os.path.join(frames_root, subclip)
        
        if os.path.isdir(frames_dir) and os.path.isdir(labels_root):
            output_path = os.path.join(output_root, f'{subclip}_ball_trajectory.mp4')
            print(f'Processing subclip: {subclip}')
            create_ball_trajectory_video(frames_dir, labels_root, output_path)
        
# Usage example
if __name__ == "__main__":
    dataset_root = "/Users/cosmincojocaru/playground/ball/"  # Replace with your dataset path
    output_root = "./test_data1"         # Replace with your desired output path
    
    process_all_subclips(dataset_root, output_root)