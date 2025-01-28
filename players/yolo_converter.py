import os
import json
from pathlib import Path
import shutil
from PIL import Image

class YOLOConverter:
    def __init__(self):
        # Class mapping
        self.class_map = {
            'ball': 0,
        }
        
    def convert_bbox_to_yolo(self, points, img_width, img_height):
        """
        Convert LabelMe points [[x1,y1], [x2,y2]] to YOLO format (x_center, y_center, width, height)
        All values are normalized between 0 and 1
        """
        x1, y1 = points[0]  # top-left
        x2, y2 = points[1]  # bottom-right
        
        # Calculate center points and dimensions
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        # Ensure values are between 0 and 1
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        width = min(max(width, 0), 1)
        height = min(max(height, 0), 1)
        
        return x_center, y_center, width, height
    
    def process_dataset(self, input_dir, output_dir):
        """
        Process the entire dataset
        """
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        for split in ['train', 'valid']:
            # Create directories
            split_img_dir = os.path.join(output_dir, split, 'images')
            split_label_dir = os.path.join(output_dir, split, 'labels')
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_label_dir, exist_ok=True)
            
            # Process files
            input_split_dir = os.path.join(input_dir, split)
            image_files = [f for f in os.listdir(input_split_dir) if f.endswith('.png')]
            
            for img_file in image_files:
                # Get corresponding JSON file
                json_file = img_file.replace('.png', '.json')
                json_path = os.path.join(input_split_dir, json_file)
                
                if not os.path.exists(json_path):
                    print(f"Warning: No JSON file found for {img_file}")
                    continue
                
                # Copy image file
                shutil.copy2(
                    os.path.join(input_split_dir, img_file),
                    os.path.join(split_img_dir, img_file)
                )
                
                # Process annotations
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Get image dimensions from JSON
                img_width = data['imageWidth']
                img_height = data['imageHeight']
                
                # Create YOLO format label file
                label_file = img_file.replace('.png', '.txt')
                label_path = os.path.join(split_label_dir, label_file)
                
                with open(label_path, 'w') as f:
                    # Process each shape
                    for shape in data['shapes']:
                        # Only process rectangle shapes with valid labels
                        if (shape['shape_type'] == 'rectangle' and 
                            shape['label'] == 'ball'):
                            
                            class_id = self.class_map[shape['label']]
                            points = shape['points']
                            
                            # Convert to YOLO format
                            x_center, y_center, width, height = self.convert_bbox_to_yolo(
                                points, img_width, img_height
                            )
                            
                            # Write to file
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Create dataset.yaml file
        yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

names:
  0: ball

nc: 1
        """
        
        with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content.strip())
        
        print(f"Dataset processed and saved to {output_dir}")
        print("Created dataset.yaml file with class mappings")

# Example usage
if __name__ == "__main__":
    converter = YOLOConverter()
    converter.process_dataset(
        input_dir="../dataset",
        output_dir="./yolo_dataset"
    )