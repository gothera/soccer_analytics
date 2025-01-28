import cv2
import random

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)

    frame_number = 0
    start_frame = 300
    paused = False  # Flag to track pause state
    frames_to_sample = sorted(random.sample(range(start_frame, 8500), 30))
    while True:
        if not paused:  # Only read new frame if not paused
            ret, frame = cap.read()
            if not ret:
                break
        frame_number += 1
        print(frame_number)
        if frame_number in frames_to_sample:    
            cv2.imwrite(f'./nba_players_dataset/4_frame_{frame_number}.jpg', frame)
        if frame_number > frames_to_sample[-1]:
            break
    cap.release()
    cv2.destroyAllWindows()
# Usage
video_path = "/Users/cosmincojocaru/Downloads/Jokić's UNREAL TRIPLE-DOUBLE vs Kings - 5 Straight Triple-Doubles! ｜ January 23, 2025.mp4"  # Replace with your video path
process_video(video_path)