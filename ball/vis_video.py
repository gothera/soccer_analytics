import cv2
import numpy as np
import os
from pathlib import Path

def visualize_ball_trajectory(clip_path, output_path):
    """
    Creates a video showing ball trajectory by drawing circles at ball positions.
    
    Args:
        clip_path (str): Path to the clip directory containing 'images' and 'labels' folders
        output_path (str): Path where the output video will be saved
    """
    # Ensure paths exist
    images_dir = Path(clip_path)
    labels_dir = Path(clip_path) / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Images or labels directory not found in {clip_path}")
    
    # Get sorted lists of image and label files
    image_files = sorted(list(images_dir.glob('*.jpg')))
    label_files = sorted(list(labels_dir.glob('*.txt')))
    
    if len(image_files) != 300 or len(label_files) != 300:
        raise ValueError(f"Expected 300 frames, but found {len(image_files)} images and {len(label_files)} labels")
    
    # Initialize video writer
    frame = cv2.imread(str(image_files[0]))
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))
    
    # Process each frame
    for img_path, label_path in zip(image_files, label_files):
        # Read image
        frame = cv2.imread(str(img_path))
        
        # Read annotation
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:  # Check if annotation exists
                # Parse YOLO format (class x_center y_center width height)
                _, x_center, y_center, _, _ = map(float, line.split())
                
                # Convert normalized coordinates to pixel coordinates
                x_pixel = int(x_center * width)
                y_pixel = int(y_center * height)
                
                # Draw circle at ball position
                cv2.circle(frame, (x_pixel, y_pixel), 2, (0, 0, 255), -1)  # Red circle
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()

def process_clip(base_dir, output_dir):
    """
    Process all clips in the base directory.
    
    Args:
        base_dir (str): Base directory containing clip_0, clip_1, etc.
        output_dir (str): Directory where output videos will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each clip directory
    clip_dirs = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('clip_')]
    clip_dir = clip_dirs[4]
    print(f"Processing {clip_dir.name}...")
    output_path = os.path.join(output_dir, f"{clip_dir.name}_annotated.mp4")
    visualize_ball_trajectory(str(clip_dir), output_path)
    print(f"Saved to {output_path}")

# Example usage
if __name__ == "__main__":
    base_directory = "./liga_1_dataset_720"  # Replace with your actual path
    output_directory = "./"     # Replace with your desired output path
    
    # Process a single clip
    # visualize_ball_trajectory("path/to/clip_0", "path/to/output/clip_0_annotated.mp4")
    
    # Or process all clips
    process_clip(base_directory, output_directory)