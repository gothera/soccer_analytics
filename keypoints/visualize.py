import os
import cv2
import numpy as np

ann_label_to_id = {
    '0': '5',
    '1': '6',
    '2': '4',
    '3': '45',
    '4': '44',
    '5': '48',
    '6': '55',
    '7': '52',
    '8': '51',
    '9': '17',
    '10': '19',
    '11': '16',
    '12': '18',
    '13': '21',
    '14': '20',
    '15': '23',
    '16': '22',
    '17': '12',
    '18': '13',
    '19': '29',
    '20': '28',
    '21': '14',
    '22': '15',
    '23': '42',
    '24': '40',
    '25': '41',
    '26': '39',
    '27': '38',
    '28': '11',
    '29': '9',
    '30': '10',
    '31': '8',
    '32': '7'
}

def load_and_visualize_keypoints(image_dir, label_dir):
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_file in image_files:
        # Read image
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Get corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"No label file found for: {img_file}")
            continue
            
        height, width = image.shape[:2]
        
        # Read annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # Split the line into values
            values = list(map(float, line.strip().split()))
            
            # Extract class index and bounding box
            class_id = int(values[0])
            x_center, y_center = values[1] * width, values[2] * height
            bbox_width, bbox_height = values[3] * width, values[4] * height
            # Calculate bbox coordinates
            x1 = int(x_center - bbox_width/2)
            y1 = int(y_center - bbox_height/2)
            x2 = int(x_center + bbox_width/2)
            y2 = int(y_center + bbox_height/2)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract keypoints
            # Starting from index 5, every 3 values represent x, y, visibility
            keypoints = []
            for i in range(5, len(values), 3):
                kp_x = int(values[i] * width)
                kp_y = int(values[i + 1] * height)
                visibility = values[i + 2]
                keypoints.append((kp_x, kp_y, visibility))
            
            # Draw keypoints
            for idx, (x, y, v) in enumerate(keypoints):
                if v > 0:  # Visible keypoint
                    color = (0, 0, 255) if v == 1 else (0, 255, 255)  # Red for v=1, Yellow for v=2
                    cv2.circle(image, (x, y), 4, color, -1)
                    cv2.putText(image, str(idx), (x + 5, y + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display image
        cv2.imshow('Keypoints Visualization', image)
        
        # Wait for key press, 'q' to quit, any other key for next image
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Directory paths
train_dir = "/Users/cosmincojocaru/playground/keypoints/keypoints_dataset/yolov8_keypoints_dataset/train"  # Replace with your train directory path
image_dir = os.path.join(train_dir, "images")
label_dir = os.path.join(train_dir, "labels")

# Run visualization
load_and_visualize_keypoints(image_dir, label_dir)