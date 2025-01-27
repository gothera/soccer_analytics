import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def process_video(model_path, video_path, output_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Increment frame counter
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        # print(results[0])

        # Process each detection
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data[0]
                print(keypoints)
                # Draw keypoints
                for kp in keypoints:
                    x, y, conf = kp
                    if conf > 0.5:  # Confidence threshold
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
                # Draw connections between keypoints (skeleton)
                # skeleton = [  # Define the connections between keypoints
                #     [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                #     [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                #     [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # ]
                
                # for connection in skeleton:
                #     kp1, kp2 = connection
                #     if keypoints[kp1-1][2] > 0.5 and keypoints[kp2-1][2] > 0.5:
                #         pt1 = (int(keypoints[kp1-1][0]), int(keypoints[kp1-1][1]))
                #         pt2 = (int(keypoints[kp2-1][0]), int(keypoints[kp2-1][1]))
                #         cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Write the frame
        out.write(frame)
        
        # Optional: Display the frame (comment out for faster processing)
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # Define paths
    model_path = '/Users/cosmincojocaru/playground/keypoints/best_yolo_keypoints.pt'  # Replace with your model path
    video_path = '/Users/cosmincojocaru/playground/players/subclip_720/subclip_12593/subclip_12593.mp4'  # Replace with your video path
    output_path = 'output_video.mp4'
    
    # Process the video
    process_video(model_path, video_path, output_path)

if __name__ == "__main__":
    main()