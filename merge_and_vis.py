import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def merge_datasets(base_dir, num_clips=14, output_dir="merged_dataset"):
    """
    Merge multiple YOLO format datasets into one.
    
    Args:
        base_dir (str): Base directory containing clip_X_annotations folders
        num_clips (int): Number of clips to merge
        output_dir (str): Output directory name
    """
    # Create output directory structure
    output_path = Path(output_dir)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    
    # Copy and rename files
    frame_counter = 0
    for clip_idx in range(num_clips):
        clip_dir = Path(base_dir) / f"clip_{clip_idx}_annotations"
        label_dir = clip_dir / "labels" / "train"
        image_dir = Path(f"./extracted_clips/clip_{clip_idx}")
        
        # Process all txt files in the clip
        for txt_file in sorted(label_dir.glob("*.txt")):
            # Copy and rename label file
            new_label_name = f"{frame_counter:06d}.txt"
            shutil.copy2(
                txt_file,
                output_path / "labels" / "train" / new_label_name
            )
            
            # Copy and rename corresponding image file
            img_file = image_dir / f"{txt_file.stem}.jpg"  # Assuming jpg format
            if img_file.exists():
                new_img_name = f"{frame_counter:06d}.jpg"
                shutil.copy2(
                    img_file,
                    output_path / "images" / "train" / new_img_name
                )
            
            frame_counter += 1
    
    # Create data.yaml in output directory
    yaml_content = """
names:
  0: ball
path: .
train: train.txt
"""
    with open(output_path / "data.yaml", "w") as f:
        f.write(yaml_content.strip())
    
    print(f"Merged dataset created with {frame_counter} frames")

# Example usage:
if __name__ == "__main__":
    # Merge datasets
    merge_datasets("./Ball_Annotations-YOLOV8")
