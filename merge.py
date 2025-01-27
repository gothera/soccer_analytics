import cv2
import numpy as np

def combine_annotated_videos(video1_path, video2_path, output_path):
    # Open both video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Check if videos are opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos")
        return
    
    # Get video properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        f1 = frame1.astype(np.float32)
        f2 = frame2.astype(np.float32)
        
        # Create masks for non-black pixels
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        mask1 = gray1 > 10
        mask2 = gray2 > 10
        
        # Blend where both frames have annotations
        overlap_mask = mask1 & mask2
        combined = f1.copy()
        
        # Copy non-overlapping annotations as is
        combined[mask2 & ~overlap_mask] = f2[mask2 & ~overlap_mask]
        
        # Blend overlapping regions
        combined[overlap_mask] = (f1[overlap_mask] + f2[overlap_mask]) / 2
        
        # Write the frame
        out.write(np.clip(combined, 0, 255).astype(np.uint8))
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release everything
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Video processing completed!")

# Usage example
if __name__ == "__main__":
    video1_path = "/Users/cosmincojocaru/playground/players/output_video.mp4"
    video2_path = "/Users/cosmincojocaru/Downloads/output_points1 (1).mp4"
    output_path = "./combined_video.mp4"
    
    combine_annotated_videos(video1_path, video2_path, output_path)