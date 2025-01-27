import cv2
import os
import numpy as np

def cut_video(input_file, cuts, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(input_file)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    for i, (start_time, end_time) in enumerate(cuts.items()):
        output_file = os.path.join(output_folder, f"cut_{i+1}.mp4")
        
        # Set up VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (1280, 720))

        # Set the video position to start_time
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > end_time * 1000:
                break
            frame = cv2.resize(frame, (1280, 720))
            out.write(frame)

        out.release()
        print(f"Cut {i+1} saved to {output_file}")

    cap.release()
    cv2.destroyAllWindows()
# Example usage
input_file = "./game2.mp4"
output_folder = "./game_cuts5"

cuts = {
    2:10
}

cut_video(input_file, cuts, output_folder)