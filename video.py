import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize_video_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.ion()  # Turn on interactive mode
    
    try:
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            if not ret:  # End of video
                break
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Clear previous frame
            ax.clear()
            
            # Display the frame
            ax.imshow(frame_rgb)
            ax.axis('off')  # Hide axes
            
            # Add frame number
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ax.set_title(f'Frame {frame_number}')
            
            # Update display
            plt.pause(0.033)  # Pause for ~30 FPS display
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Clean up
        cap.release()
        plt.close()

# Alternative version using FuncAnimation for smoother playback
def visualize_video_animated(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    def update(frame):
        ret, img = cap.read()
        if ret:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(img_rgb)
            ax.axis('off')
            ax.set_title(f'Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}')
    
    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create animation
    anim = FuncAnimation(fig, update, 
                        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        interval=1000/fps,  # interval in milliseconds
                        repeat=False)
    
    plt.show()
    
    # Clean up
    cap.release()

# Usage example
if __name__ == "__main__":
    video_path = "/Users/cosmincojocaru/playground/WILD OT ENDING Clippers vs Celtics Uncut ｜ January 22, 2025.mp4"
    
    # Choose one of the visualization methods:
    # Method 1: Frame by frame display
    visualize_video_frames(video_path)
    
    # Method 2: Animated display (smoother playback)
    # visualize_video_animated(video_path)