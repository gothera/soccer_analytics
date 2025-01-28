import cv2
import numpy as np
import time
import argparse
import torch
import colorsys

from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from sklearn.cluster import KMeans

class VideoProcessor:
    def __init__(
        self,
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        n_colors=1,
        device=None
    ):
        """
        Initialize the video processor
        model_path: path to trained YOLOv8 model
        conf_threshold: confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        device: device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize model
        self.model = YOLO(model_path)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Class colors (BGR format)
        self.colors = {
            0: (0, 255, 0),    # player: Green
            1: (255, 0, 0),    # keeper: Blue
            2: (0, 0, 255)     # referee: Red
        }
        
        # Class names
        self.class_names = {
            0: 'player',
            1: 'keeper',
            2: 'referee'
        }

        self.n_colors = n_colors  # Number of dominant colors to extract per player
        self.team_colors = None
        self.color_threshold = 30  # Color distance threshold for team assignment
    
    def get_dominant_colors(self, image):
        """Extract dominant colors from an image using K-means clustering."""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove very dark pixels (likely to be shadows or noise)
        mask = (pixels.mean(axis=1) > 30)
        pixels = pixels[mask]
        
        if len(pixels) == 0:
            return None
            
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors and their counts
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Sort colors by frequency
        colors = [colors[i] for i, _ in counts.most_common()]
        return colors[0] if colors else None

    def initialize_team_colors(self, frame, results):
        """Initialize team colors by clustering all player colors."""
        all_colors = []
        
        # Extract colors from all player bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Only consider players, not referees or goalkeepers
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    player_img = frame[y1:y2, x1:x2]
                    dominant_color = self.get_dominant_colors(player_img)
                    if dominant_color is not None:
                        all_colors.append(dominant_color)
        
        if not all_colors:
            return None
            
        # Cluster all colors to find two team colors
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(all_colors)
        self.team_colors = kmeans.cluster_centers_.astype(int)
        return self.team_colors
    
    def color_distance(self, color1, color2):
        """Calculate distance between two colors in HSV space."""
        # Convert BGR to RGB then to HSV
        # color1 and color2 are in BGR format from OpenCV
        hsv1 = colorsys.rgb_to_hsv(color1[2]/255, color1[1]/255, color1[0]/255)
        hsv2 = colorsys.rgb_to_hsv(color2[2]/255, color2[1]/255, color2[0]/255)
        
        # Calculate weighted distance (giving more importance to hue)
        hue_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0])) * 2
        sat_diff = abs(hsv1[1] - hsv2[1])
        val_diff = abs(hsv1[2] - hsv2[2])
        
        return hue_diff * 0.5 + sat_diff * 0.25 + val_diff * 0.25
    
    def classify_player(self, player_img):
        """Classify a player image into one of the teams."""
        if self.team_colors is None:
            return -1
            
        dominant_color = self.get_dominant_colors(player_img)
        if dominant_color is None:
            return -1
            
        # Calculate distance to each team color
        distances = [self.color_distance(dominant_color, team_color) 
                    for team_color in self.team_colors]
        
        # If the closest distance is too large, might be referee or goalkeeper
        min_distance = min(distances)
        if min_distance > self.color_threshold:
            return -1
            
        return np.argmin(distances)
    
    def draw_boxes(self, frame, classifications):
        """Draw boxes with team-specific colors."""
        output_frame = frame.copy()
        
        # Define colors for teams and special roles
        team_bbox_colors = [(0, 0, 255), (255, 0, 0)]  # Red and Blue for teams
        special_colors = {-1: (0, 255, 0)}  # Green for referee/unclassified
        
        for clf in classifications:
            box = clf['box']
            team = clf['team']
            cls = clf['class']
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color = team_bbox_colors[team] if team >= 0 else special_colors[-1]  
            
            # Calculate triangle parameters
            triangle_width = 20  # Width of the triangle base
            triangle_height = 15  # Height of the triangle
            vertical_offset = 30  # Offset above the bounding box
            
            # Calculate triangle points
            center_x = (x1 + x2) // 2
            triangle_base_y = y1 - vertical_offset  # Base of triangle (now at top)
            
            # Define three points of the triangle (inverted)
            triangle_points = np.array([
                [center_x - triangle_width//2, triangle_base_y],  # Top left
                [center_x + triangle_width//2, triangle_base_y],  # Top right
                [center_x, triangle_base_y + triangle_height]  # Bottom center (pointing down)
            ], np.int32)
            
            # Draw filled triangle
            cv2.fillPoly(output_frame, [triangle_points], color)
            
        return output_frame

    def process_frame(self, frame, results):
        """Process a frame and classify all players."""
        # Initialize team colors if not already done
        if self.team_colors is None:
            self.initialize_team_colors(frame, results)
            
        classifications = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                player_img = frame[y1:y2, x1:x2]
                
                # if cls == 0:  # Player
                #     team = self.classify_player(player_img)
                # else:  # Referee or goalkeeper
                #     team = -1
                    
                classifications.append({
                    'box': box,
                    'team': -1,
                    'class': cls
                })
                
        return classifications
    
    def process_video(
        self,
        input_path,
        output_path,
        end_time=None,
        show_progress=True,
        display=False
    ):
        """
        Process video file and create output with detections
        """
        # Open video file
        cap = cv2.VideoCapture(input_path)
        cap_output = cv2.VideoCapture('/Users/cosmincojocaru/playground/merged_output.mp4')
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up video writer
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (1280, 720))        
        try:
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                ret_output, output_frame = cap_output.read()
                if not ret or not ret_output:
                    break

                # Run inference
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device
                )

                if self.team_colors is None:
                    self.initialize_team_colors(frame, results)
                classifications = self.process_frame(frame, results)
                # Draw detections
                annotated_frame = self.draw_boxes(output_frame, classifications)
                # Write frame
                out.write(annotated_frame)    
        finally:
            # Clean up
            cap.release()
            out.release()
            cap_output.release()
            cv2.destroyAllWindows()
            
        # Print statistics
        print(f"Output saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with YOLOv8 model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, default='./output_nba_players_ball.mp4',help='Output video path')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, or None for auto)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    # Process video
    processor.process_video(
        input_path=args.input,
        output_path=args.output,
    )