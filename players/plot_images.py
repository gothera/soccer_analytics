import cv2
import os
import numpy as np

class YOLOVisualizer:
    def __init__(self, dataset_yaml):
        # Read dataset.yaml
        with open(dataset_yaml, 'r') as f:
            # Basic YAML parsing without requiring yaml package
            lines = f.readlines()
            self.base_path = None
            self.class_names = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('path:'):
                    self.base_path = line.split('path:')[1].strip()
                elif line.startswith('train:'):
                    self.train_path = line.split('train:')[1].strip()
                elif line.startswith('val:'):
                    self.val_path = line.split('val:')[1].strip()
                elif ':' in line and 'names' not in line:
                    try:
                        idx, name = line.split(':')
                        self.class_names[int(idx)] = name.strip()
                    except:
                        continue

        # Colors for different classes (BGR format)
        self.colors = {
            0: (0, 255, 0),    # player: Green
            1: (255, 0, 0),    # keeper: Blue
            2: (0, 0, 255)     # referee: Red
        }
        
        # Window name
        self.window_name = "YOLO Dataset Visualization"
        
    def convert_yolo_to_bbox(self, x_center, y_center, width, height, img_width, img_height):
        """
        Convert YOLO format (x_center, y_center, width, height) to bbox (x1, y1, x2, y2)
        """
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        return x1, y1, x2, y2
    
    def draw_boxes(self, img, labels, img_width, img_height):
        """
        Draw bounding boxes and labels on the image
        """
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            class_id = int(class_id)
            
            # Convert YOLO coordinates to pixel coordinates
            x1, y1, x2, y2 = self.convert_yolo_to_bbox(
                x_center, y_center, width, height, img_width, img_height
            )
            
            # Draw rectangle
            color = self.colors.get(class_id, (255, 255, 255))  # default to white if class not found
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            label = f"{class_name}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # cv2.rectangle(img, (x1, y1 - label_height - 5), (x1 + label_width + 5, y1), color, -1)
            # cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return img
    
    def visualize_dataset(self, split='train'):
        """
        Visualize images and annotations from the specified split
        """
        # Get paths
        if split == 'train':
            images_path = os.path.join(self.base_path, self.train_path)
        else:
            images_path = os.path.join(self.base_path, self.val_path)
            
        labels_path = images_path.replace('images', 'labels')
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"No images found in {images_path}")
            return
        
        current_idx = 0
        paused = True
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while True:
            img_file = image_files[current_idx]
            label_file = os.path.splitext(img_file)[0] + '.txt'
            
            # Read image
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Error reading image: {img_path}")
                current_idx = (current_idx + 1) % len(image_files)
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Read and draw labels
            label_path = os.path.join(labels_path, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.readlines()
                img = self.draw_boxes(img, labels, img_width, img_height)
            
            # Draw image info
            info_text = f"{split}: {current_idx + 1}/{len(image_files)} - {img_file}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show image
            cv2.imshow(self.window_name, img)
            
            # Handle keyboard input
            if paused:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(100)
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('n') or key == ord('d'):  # Next image
                current_idx = (current_idx + 1) % len(image_files)
            elif key == ord('p') or key == ord('a'):  # Previous image
                current_idx = (current_idx - 1) % len(image_files)
            elif key == 32:  # Space bar - toggle pause
                paused = not paused
            elif key == ord('s'):  # Save current image
                output_path = f"visualization_{img_file}"
                cv2.imwrite(output_path, img)
                print(f"Saved visualization to: {output_path}")
        
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    visualizer = YOLOVisualizer("./liga_1_players_dataset/data.yaml")
    
    # Visualize training set
    print("Visualizing training set...")
    print("Controls:")
    print("  - Space: Toggle pause/play")
    print("  - N or D: Next image")
    print("  - P or A: Previous image")
    print("  - S: Save current visualization")
    print("  - Q: Quit")
    visualizer.visualize_dataset(split='train')
    
    # Visualize validation set
    print("\nVisualizing validation set...")
    visualizer.visualize_dataset(split='val')