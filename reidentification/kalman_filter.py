import cv2
import numpy as np
from ultralytics import YOLO
import argparse

class KalmanPlayerTracker:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.tracking_player = False
        self.selected_box = None
        self.output_video = None
        
        # Kalman Filter with 8 state variables [x, y, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x += vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y += vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w += vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h += vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix - we only measure position and size
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Increase process noise for velocity components
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.1  # Velocity components
        self.kf.processNoiseCov[:4, :4] *= 0.01  # Position components
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Initial state covariance
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        self.last_prediction = None
        self.frames_since_detection = 0
        self.MAX_FRAMES_LOST = 30
        
    def box_to_measurement(self, box):
        """Convert [x1, y1, x2, y2] box to [x, y, w, h] measurement"""
        x = (box[0] + box[2]) / 2  # center x
        y = (box[1] + box[3]) / 2  # center y
        w = box[2] - box[0]        # width
        h = box[3] - box[1]        # height
        return np.array([[x], [y], [w], [h]], dtype=np.float32)
    
    def measurement_to_box(self, measurement):
        """Convert [x, y, w, h] measurement to [x1, y1, x2, y2] box"""
        x, y, w, h = measurement.flatten()
        return np.array([
            x - w/2,  # x1
            y - h/2,  # y1
            x + w/2,  # x2
            y + h/2   # y2
        ], dtype=np.float32)

    def initialize_kalman_filter(self, box):
        """Initialize Kalman filter state with first detection"""
        measurement = self.box_to_measurement(box)
        
        # Initialize state [x, y, w, h, vx=0, vy=0, vw=0, vh=0]
        state = np.zeros((8, 1), dtype=np.float32)
        state[:4] = measurement  # First 4 elements are the measurement
        self.kf.statePost = state
        
        # Initialize velocities with low uncertainty
        self.kf.errorCovPost[4:, 4:] *= 0.1

    def predict_and_update(self, detections):
        """Predict new state and update with measurement if available"""
        # Predict
        prediction = self.kf.predict()
        predicted_box = self.measurement_to_box(prediction[:4])
        self.last_prediction = predicted_box
        
        # Find best matching detection using IoU
        best_iou = 0
        best_detection = None
        
        for detection in detections:
            iou = self.calculate_iou(predicted_box, detection)
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_detection = detection
        
        # Update phase
        if best_detection is not None:
            self.frames_since_detection = 0
            measurement = self.box_to_measurement(best_detection)
            self.kf.correct(measurement)
            return self.measurement_to_box(self.kf.statePost[:4])
        else:
            self.frames_since_detection += 1
            return predicted_box if self.frames_since_detection < self.MAX_FRAMES_LOST else None
    
    def calculate_iou(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select a player"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.tracking_player:
            success, frame = self.cap.read()
            if not success:
                return
                
            results = self.model(frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                if (x > box[0] and x < box[2] and 
                    y > box[1] and y < box[3]):
                    self.selected_box = box
                    self.initialize_kalman_filter(box)
                    self.tracking_player = True
                    self.setup_output_video(frame)
                    break

    def setup_output_video(self, frame):
        """Initialize the output video writer"""
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(
            'tracked_player_kalman1.mp4',
            fourcc,
            50.0,
            (width, height)
        )

    def track_player(self):
        """Main tracking loop"""
        cv2.namedWindow('Player Tracking')
        cv2.setMouseCallback('Player Tracking', self.mouse_callback)

        frame_idx = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            frame_idx += 1
            if frame_idx < 26000:
                continue
            results = self.model(frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()

            # Draw all detected players in green
            for box in boxes:
                cv2.rectangle(frame, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 2)

            # Update tracking if we're following a player
            if self.tracking_player:
                tracked_box = self.predict_and_update(boxes)
                if tracked_box is not None:
                    # Draw tracked player in red
                    cv2.rectangle(frame,
                                (int(tracked_box[0]), int(tracked_box[1])),
                                (int(tracked_box[2]), int(tracked_box[3])),
                                (0, 0, 255), 3)
                    
                    # Draw Kalman prediction in blue
                    if self.last_prediction is not None:
                        cv2.rectangle(frame,
                                    (int(self.last_prediction[0]), int(self.last_prediction[1])),
                                    (int(self.last_prediction[2]), int(self.last_prediction[3])),
                                    (255, 0, 0), 2)

            cv2.imshow('Player Tracking', frame)
            
            if self.tracking_player and self.output_video is not None:
                self.output_video.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if self.output_video is not None:
            self.output_video.release()
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Soccer Player Tracking Tool with Kalman Filter')
    parser.add_argument('--video_path', type=str, default='./game_1080.mp4', required=False, help='Path to the input video file')
    parser.add_argument('--model_path', type=str, default='../players/best_players.pt', required=False, help='Path to the YOLOv8 model weights')
    args = parser.parse_args()

    tracker = KalmanPlayerTracker(args.video_path, args.model_path)
    tracker.track_player()

if __name__ == "__main__":
    main()