from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

def process_video(model_path, input_video_path, output_video_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Open the video file
    video = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        
        if not success:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
        
        # Run inference on the frame
        results = model(frame, conf=0.25)  # Adjust confidence threshold as needed
        
        # Draw the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to the output video
        out.write(annotated_frame)
        
        # Optional: Display the frame (comment out for faster processing)
        # cv2.imshow('YOLOv8 Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = "./best_nba_players.pt"
    input_video = "/Users/cosmincojocaru/playground/merged_output.mp4"  # Replace with your input video path
    output_video = "output_nba_video.mp4"  # Replace with desired output path
    
    process_video(model_path, input_video, output_video)