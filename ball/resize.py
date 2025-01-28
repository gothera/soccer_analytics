import os
import cv2
from pathlib import Path
from tqdm import tqdm

def resize_frames(input_root, output_root, target_size=(1280, 720)):
    """
    Resize all frames in the dataset to the specified target size.
    
    Args:
        input_root (str): Path to liga_1_ball_dataset
        output_root (str): Path where resized frames will be saved
        target_size (tuple): Target resolution as (width, height)
    """
    # Create output root directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    
    # Get list of all subclip directories
    subclips = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for subclip in subclips:
        subclip_dir = os.path.join(input_root, subclip)
        if not os.path.isdir(os.path.join(output_root, subclip)):
            os.makedirs(os.path.join(output_root, subclip), exist_ok=True)

        for f in os.listdir(subclip_dir):
            if not f.endswith('.jpg'):
                continue
            # print(f)
            input_frame_path = os.path.join(input_root, subclip, f)
            output_frame_path = os.path.join(output_root, subclip, f)
            
            print(input_frame_path, output_frame_path)
            # Read frame
            frame = cv2.imread(input_frame_path)
            
            # Resize frame
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            
            # Save resized frame
            cv2.imwrite(output_frame_path, resized_frame)

if __name__ == "__main__":
    # Replace these paths with your actual paths
    input_dataset = "/Users/cosmincojocaru/playground/ball/nba_dataset"
    output_dataset = "/Users/cosmincojocaru/playground/ball/nba_dataset_720"
    
    # Run the resize operation
    resize_frames(input_dataset, output_dataset)
    print("Resizing completed!")