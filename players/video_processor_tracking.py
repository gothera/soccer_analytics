from ultralytics import YOLO
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import torch
from collections import defaultdict
from utils import *
import supervision as sv

def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image

class VideoProcessor:
    def __init__(
        self,
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device=None,
        track_buffer=25,  # Track buffer for BYTETracker
        track_thresh=0.5,  # Detection confidence threshold
        match_thresh=0.8,  # IOU threshold for matching
        frame_rate=25      # Video frame rate for motion model
    ):
        """
        Initialize the video processor with tracking capabilities
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
        
        self.tracker = BYTETracker(BYTETrackerArgs())


        # self.annotator = BaseAnnotator(
        #     colors=COLORS, 
        #     thickness=4)
        self.annotator = TextAnnotator(background_color=Color(255, 255, 255), text_color=Color(0, 0, 0), text_thickness=2)


        # self.box_annotator = sv.BoxAnnotator()
        # self.label_annotator = sv.LabelAnnotator()
        # Track history for visualization
        self.track_history = defaultdict(lambda: [])
        
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
    
    def process_video(
        self,
        input_path,
        output_path,
        end_time=None,
        show_progress=True,
        display=False
    ):
        """
        Process video file with tracking
        """
        cap = cv2.VideoCapture(input_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO detection
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device
                )[0]
                

                # detections = []
                # for (x_min, y_min, x_max, y_max), confidence, class_id in zip(list(results.boxes.xyxy.cpu().numpy()), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                #     class_id=int(class_id)
                #     detections.append(Detection(
                #         rect=Rect(
                #             x=float(x_min),
                #             y=float(y_min),
                #             width=float(x_max - x_min),
                #             height=float(y_max - y_min)
                #         ),
                #         class_id=class_id,
                #         class_name=self.class_names[class_id],
                #         confidence=float(confidence)
                #     ))

                # tracks = self.tracker.update(
                #     output_results=detections2boxes(detections=detections),
                #     img_info=frame.shape,
                #     img_size=frame.shape
                # )
                # player_detections = match_detections_with_tracks(detections=detections, tracks=tracks)
                
                annotated_frame = frame.copy()
                # print("Da", len(labels), len(results.boxes))
                for x_min, y_min, x_max, y_max in results.boxes.xyxy.cpu().numpy():
                    rect=Rect(
                        x=float(x_min),
                        y=float(y_min),
                        width=float(x_max - x_min),
                        height=float(y_max - y_min)
                    )
                    annotated_frame = draw_ellipse(annotated_frame, rect=rect, color=COLORS[0])
                # Write frame
                # annotate video frame
                # annotated_image = frame.copy()
                # annotated_image = self.annotator.annotate(
                #     image=annotated_image, 
                #     detections=detections)
                out.write(annotated_frame)
                
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
        print(f"Output saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with YOLOv8 model and BYTETracker')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='./output_1.mp4',help='Output video path')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, or None for auto)')
    parser.add_argument('--display', action='store_true', help='Display output while processing')
    
    args = parser.parse_args()
    
    processor = VideoProcessor(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    processor.process_video(
        input_path=args.input,
        output_path=args.output,
        display=args.display
    )