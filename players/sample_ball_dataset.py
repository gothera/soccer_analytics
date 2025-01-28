import cv2
import random

def generate_uniform_spaced_numbers(start, end, count, min_difference):
    # Validate inputs
    available_range = end - start
    required_range = (count - 1) * min_difference
    
    if required_range > available_range:
        raise ValueError("Range too small for given count and minimum difference")
    
    # Calculate section size for uniform distribution
    section_size = available_range / count
    
    result = []
    for i in range(count):
        # Calculate bounds for this section
        section_start = start + int(i * section_size)
        section_end = start + int((i + 1) * section_size)
        
        # Adjust bounds to respect minimum difference
        if result and section_start < result[-1] + min_difference:
            section_start = result[-1] + min_difference
            
        if i < count - 1:  # If not the last number
            section_end = min(section_end, end - (count - i - 1) * min_difference)
        
        # If valid range exists for this section
        if section_start <= section_end:
            number = random.randint(section_start, section_end)
            result.append(number)
        else:
            raise ValueError("Cannot maintain uniform distribution with given constraints")
    
    return result

def process_video(video_path):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps is", fps)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_number, idx = 0, 0
    start_frame, end_frame, window_size = 550, 8500, 60
    paused = False  # Flag to track pause state
    frames_to_sample = generate_uniform_spaced_numbers(start_frame, end_frame, 10, 100)
    # print(frames_to_sample)
    while True:
        if not paused:  # Only read new frame if not paused
            ret, frame = cap.read()
            if not ret:
                break
        frame_number += 1
        # print(frame_number)
        if frame_number < start_frame:
            continue
        if idx == len(frames_to_sample):
            break
        if frame_number == frames_to_sample[idx]:
            cv2.imwrite(f'./nba_5/nba_5_frame_{frame_number}_1.jpg', frame)
        if frame_number > frames_to_sample[idx] and frame_number < frames_to_sample[idx] + window_size:
            cv2.imwrite(f'./nba_5/nba_5_frame_{frame_number}.jpg', frame)
            # print("saved to ", f'./liga_1_ball_dataset/frame_{frame_number}.jpg')
        if frame_number == frames_to_sample[idx] + window_size:
            idx += 1        
    cap.release()
    cv2.destroyAllWindows()
# Usage
video_path = "/Users/cosmincojocaru/Downloads/Jokić's UNREAL TRIPLE-DOUBLE vs Kings - 5 Straight Triple-Doubles! ｜ January 23, 2025.mp4"  # Replace with your video path
process_video(video_path)