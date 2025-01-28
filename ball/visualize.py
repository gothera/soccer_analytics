import os
import random
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

class DatasetVisualizer:
    def __init__(self, base_dir: str, num_clips: int = 14):
        """
        Initialize the visualizer with dataset path and number of clips
        """
        self.base_dir = base_dir
        self.num_clips = num_clips
        
    def load_random_frame(self) -> tuple:
        """
        Load a random frame and its corresponding ground truth
        Returns: (frame, gt_image, frame_info)
        """
        # Select random clip
        clip_idx = random.randint(0, self.num_clips - 1)
        clip_name = f"clip_{clip_idx}" if clip_idx > 0 else "clip"
        
        # Get all frames in the clip
        frames_dir = os.path.join(self.base_dir, clip_name, "images")
        gt_dir = os.path.join(self.base_dir, clip_name, "gts")
        
        frame_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
            
        # Select random frame
        frame_path = random.choice(frame_files)
        frame_name = os.path.basename(frame_path)
        gt_path = os.path.join(gt_dir, frame_name.replace('.jpg', '.png'))
        
        # Load images
        frame = cv2.imread(frame_path)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if gt_image is None:
            raise ValueError(f"Ground truth image not found: {gt_path}")
        
        # Find ball position from ground truth
        ball_y, ball_x = np.where(gt_image == 255)
        ball_pos = (ball_x[0], ball_y[0]) if len(ball_x) > 0 else None
        
        frame_info = {
            'clip': clip_name,
            'frame': frame_name,
            'ball_position': ball_pos
        }
        
        return frame, gt_image, frame_info
    
    def create_overlay(self, frame: np.ndarray, gt_image: np.ndarray, 
                      alpha: float = 0.7, gt_color: tuple = (0, 255, 0)) -> np.ndarray:
        """
        Create overlay of ground truth on original frame
        """
        # Create a colored version of the ground truth
        # gt_colored = np.zeros_like(frame)
        # gt_colored[gt_image == 255] = gt_color
        
        # Blend the images
        alpha = 0.4
        # Normalize heatmap to 0-255 range if not already
        if gt_image.max() <= 1.0:
            gt_image = (gt_image * 255).astype(np.uint8)
        else:
            gt_image = gt_image.astype(np.uint8)
        
        # Apply colormap to create RGB heatmap
        heatmap_colored = cv2.applyColorMap(gt_image, cv2.COLORMAP_JET)
    
        overlay = cv2.addWeighted(frame, alpha, heatmap_colored, 1 - alpha, 0)
        
        return overlay
    
    def display_frame(self, num_frames: int = 1, figsize: tuple = (15, 5)):
        """
        Display random frames with ground truth overlay
        """
        plt.figure(figsize=figsize)
        
        for i in range(num_frames):
            plt.subplot(1, num_frames, i + 1)
            
            try:
                frame, gt_image, frame_info = self.load_random_frame()
                
                # Create overlay
                overlay = self.create_overlay(frame, gt_image)
                
                # Convert BGR to RGB for matplotlib
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                plt.imshow(overlay)
                plt.axis('off')
                
                # Add title with frame info
                ball_pos = frame_info['ball_position']
                title = f"{frame_info['clip']}/{frame_info['frame']}"
                if ball_pos:
                    title += f"\nBall at ({ball_pos[0]}, {ball_pos[1]})"
                plt.title(title)
                
            except Exception as e:
                print(f"Error loading frame: {e}")
                plt.text(0.5, 0.5, 'Error loading frame', 
                        horizontalalignment='center',
                        verticalalignment='center')
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize dataset with ground truth overlays')
    parser.add_argument('--dataset_path', type=str, default='./liga_1_dataset_720',
                       help='Path to the dataset base directory')
    parser.add_argument('--num_frames', type=int, default=3,
                       help='Number of random frames to display')
    parser.add_argument('--num_clips', type=int, default=14,
                       help='Number of clips in the dataset')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Alpha value for overlay blending (0.0 to 1.0)')
    parser.add_argument('--gt_color', type=str, default='green',
                       choices=['red', 'green', 'blue', 'yellow'],
                       help='Color for ground truth overlay')
    
    args = parser.parse_args()
    
    # Convert color name to BGR values
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255)
    }
    
    visualizer = DatasetVisualizer(args.dataset_path, args.num_clips)
    
    while True:
        visualizer.display_frame(
            num_frames=args.num_frames,
        )
        
        # Ask if user wants to continue
        response = input("\nPress Enter to see more frames, or 'q' to quit: ")
        if response.lower() == 'q':
            break

if __name__ == "__main__":
    main()