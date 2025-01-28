import os
import shutil
import cv2
import glob
import re

def extract_frame_number(filename):
    # Extract number from filenames like "frame_5643_1.jpg"
    match = re.match(r'nba_5_frame_(\d+)(?:_1)?\.jpg', filename)
    if match:
        return int(match.group(1))
    return -1

def organize_and_create_subclips(input_dir, fps=30.0):
    # Get all frames and sort them
    all_frames = sorted(glob.glob(os.path.join(input_dir, '*.jpg')), 
                       key=lambda x: extract_frame_number(os.path.basename(x)))
    
    if not all_frames:
        print("No frames found in directory")
        return
    
    # Find starting frames (ending with _1)
    clip_starts = [f for f in all_frames if f.endswith('_1.jpg')]
    
    if not clip_starts:
        print("No clip start frames (_1) found")
        return
    
    # Process each subclip
    for i in range(len(clip_starts)):
        start_frame = clip_starts[i]
        start_num = extract_frame_number(os.path.basename(start_frame))
        
        # Determine end frame
        if i < len(clip_starts) - 1:
            end_num = extract_frame_number(os.path.basename(clip_starts[i + 1])) - 1
        else:
            end_num = extract_frame_number(os.path.basename(all_frames[-1]))
        
        # Create subclip directory
        subclip_dir = os.path.join(input_dir, f'subclip_{start_num}')
        os.makedirs(subclip_dir, exist_ok=True)
        
        print(f"\nProcessing subclip from frame {start_num} to {end_num}")
        
        # Move frames to subclip directory
        subclip_frames = []
        for frame in all_frames:
            frame_num = extract_frame_number(os.path.basename(frame))
            if start_num <= frame_num <= end_num:
                dest_path = os.path.join(subclip_dir, os.path.basename(frame))
                shutil.move(frame, dest_path)
                subclip_frames.append(dest_path)
        
        # Create video from frames
        if subclip_frames:
            # Read first frame to get dimensions
            first_frame = cv2.imread(subclip_frames[0])
            height, width = first_frame.shape[:2]
            
            # Create video writer
            output_video = os.path.join(subclip_dir, f'subclip_{start_num}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Write frames to video
            for frame_path in sorted(subclip_frames, 
                                   key=lambda x: extract_frame_number(os.path.basename(x))):
                frame = cv2.imread(frame_path)
                out.write(frame)
            
            out.release()
            print(f"Created video: {output_video}")

# Usage
input_directory = './nba_5'
organize_and_create_subclips(input_directory)