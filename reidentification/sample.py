import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk
import json
import os
from datetime import datetime
import random
import warnings

class AnnotationTool:
    def __init__(self, video_path, yolo_model_path, frame_interval=500):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(yolo_model_path)
        
        # Initialize tkinter window
        self.window = tk.Tk()
        self.window.title("Soccer Player Annotation Tool")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create canvas for displaying video frames
        self.canvas = tk.Canvas(self.main_frame, width=1920, height=1080)
        self.canvas.pack()
        
        # Control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(pady=10)
        
        self.next_btn = ttk.Button(self.control_frame, text="Next Frame", command=self.next_frame)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(self.control_frame, text="Save Annotations", command=self.save_annotations)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Initialize state variables
        self.current_frame_number = 0
        self.current_frame = None
        self.current_detections = None
        self.target_box = None
        self.collected_samples = []
        self.frames_processed = 0
        self.target_size = (128, 256)  # Width, Height - typical for person re-ID

        # Bind mouse click
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        self.update_status()

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def find_non_overlapping_boxes(self, target_box, all_boxes, num_required=10, iou_threshold=0.0):
        """Find boxes with zero IoU with the target box and each other"""
        non_overlapping_boxes = []
        candidate_boxes = [b for b in all_boxes if not np.array_equal(b, target_box)]
        
        for box in candidate_boxes:
            # Check IoU with target box
            if self.calculate_iou(target_box, box) > iou_threshold:
                continue
                
            # Check IoU with already selected boxes
            overlaps = False
            for selected_box in non_overlapping_boxes:
                if self.calculate_iou(box, selected_box) > iou_threshold:
                    overlaps = True
                    break
                    
            if not overlaps:
                non_overlapping_boxes.append(box)
                
            if len(non_overlapping_boxes) == num_required:
                break
        
        if len(non_overlapping_boxes) < num_required:
            warnings.warn(f"Could only find {len(non_overlapping_boxes)} non-overlapping boxes " 
                        f"instead of the requested {num_required}")
            
        return non_overlapping_boxes
    
    def update_status(self):
        status = f"Frames processed: {self.frames_processed}/20 | "
        status += f"Target boxes: {len([s for s in self.collected_samples if s['is_target']])} | "
        status += f"Non-target boxes: {len([s for s in self.collected_samples if not s['is_target']])}"
        self.status_var.set(status)

    def next_frame(self):
        if self.frames_processed >= 5:
            self.status_var.set("Maximum frames reached! Please save your annotations.")
            return

        # Skip frames according to interval
        for _ in range(self.frame_interval):
            ret, _ = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, _ = self.cap.read()
            self.current_frame_number += 1

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Run YOLO detection
            results = self.model(frame)
            
            # Filter for person class (typically class 0 in YOLO)
            boxes = []
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    if cls == 0 and conf > 0.5:  # Person class with confidence > 0.5
                        boxes.append(box.cpu().numpy())
            
            self.current_detections = boxes
            
            # Display frame with boxes
            self.display_frame()

    def display_frame(self):
        frame = self.current_frame.copy()

        # Draw all detection boxes
        for i, box in enumerate(self.current_detections):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert to tkinter format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = PIL.Image.fromarray(frame_rgb)
        self.photo = PIL.ImageTk.PhotoImage(image=frame_pil)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def handle_click(self, event):
        if not self.current_detections or self.frames_processed >= 20:
            return
            
        # Find if click is inside any box
        for box in self.current_detections:
            x1, y1, x2, y2 = map(int, box)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                # Store target box
                target_box = self.extract_box(self.current_frame, box)
                self.collected_samples.append({
                    'frame_number': self.current_frame_number,
                    'box': target_box,
                    'is_target': True
                })
                
                # Find non-overlapping boxes
                non_overlapping_boxes = self.find_non_overlapping_boxes(
                    box, 
                    self.current_detections,
                    num_required=10,
                    iou_threshold=0.0
                )
                
                # Store non-overlapping boxes as negative samples
                for neg_box in non_overlapping_boxes:
                    box_img = self.extract_box(self.current_frame, neg_box)
                    self.collected_samples.append({
                        'frame_number': self.current_frame_number,
                        'box': box_img,
                        'is_target': False
                    })
                
                self.frames_processed += 1
                self.update_status()
                self.next_frame()
                break
    

    def extract_box(self, frame, box):
        """
        Extract player crop from frame using bounding box
        
        Args:
            frame: numpy array of shape (H, W, C)
            box: list/tuple [x1, y1, x2, y2] 
        Returns:
            resized crop of the player
        """
        try:
            # Extract integer coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)
            
            # Check valid box size
            if x2 <= x1 or y2 <= y1:
                return None
                
            # Extract and resize crop
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
                
            resized = cv2.resize(crop, self.target_size)
            return resized
            
        except Exception as e:
            print(f"Error cropping box: {e}")
            print(f"Frame shape: {frame.shape}")
            print(f"Box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None

    def save_annotations(self):
        if not self.collected_samples:
            return
            
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"annotations_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images and create annotation file
        annotations = []
        for i, sample in enumerate(self.collected_samples):
            img_filename = f"box_{i}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            cv2.imwrite(img_path, sample['box'])
            
            annotations.append({
                'image': img_filename,
                'frame_number': sample['frame_number'],
                'is_target': sample['is_target']
            })
            
        # Save annotation file
        with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f, indent=2)
            
        self.status_var.set(f"Annotations saved to {output_dir}")

    def run(self):
        self.next_frame()
        self.window.mainloop()

if __name__ == "__main__":
    # Example usage
    VIDEO_PATH ='/Users/cosmincojocaru/Downloads/Liga.1.Dinamo.Bucuresti.vs.Sepsi.Sf.Gheorghe.30.11.2024.1080i.HDTV.MPA2.0.H.264-playTV/Liga.1.Dinamo.Bucuresti.vs.Sepsi.Sf.Gheorghe.30.11.2024.1080i.HDTV.MPA2.0.H.264-playTV.mkv'
    YOLO_MODEL_PATH = "../players/last.pt"  # or path to your custom trained model

    tool = AnnotationTool(VIDEO_PATH, YOLO_MODEL_PATH)
    tool.run()