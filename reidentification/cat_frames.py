import cv2
import numpy as np

def process_video(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    recording = True  # Flag to track if we're currently recording
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no < 1000:
            continue
        # Display the frame
        cv2.imshow('Video', frame)

        # Write frame to output video if recording is True
        if recording:
            out.write(frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        # Press spacebar to toggle recording
        elif key == ord(' '):
            recording = not recording
            status = "Recording" if recording else "Paused"
            print(f"Recording status: {status}")

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video = "/Users/cosmincojocaru/Downloads/Liga.1.Dinamo.Bucuresti.vs.Sepsi.Sf.Gheorghe.30.11.2024.1080i.HDTV.MPA2.0.H.264-playTV/Liga.1.Dinamo.Bucuresti.vs.Sepsi.Sf.Gheorghe.30.11.2024.1080i.HDTV.MPA2.0.H.264-playTV.mkv"  # Replace with your input video path
output_video = "output.mp4"  # Replace with desired output path
process_video(input_video, output_video) 