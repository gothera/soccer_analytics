import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import cv2
# Load the JSON data
with open('my_dataset/img0.json', 'r') as f:
    data = json.load(f)

# Extract the keypoints
keypoints = data['shapes'][0]['points']

# Convert keypoints to numpy array
keypoints = np.array(keypoints)

# Load the image
image_path = data['imagePath']
img = plt.imread(image_path)
img = cv2.resize(img, (640, 360))

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Display the image
ax.imshow(img)

keypoints /= 8
# Plot the keypoints
ax.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=50)

# Plot the polygon
# polygon = Polygon(keypoints, fill=False, edgecolor='r')
# ax.add_patch(polygon)

# Set the axis limits
# ax.set_xlim(0, data['imageWidth'])
# ax.set_ylim(data['imageHeight'], 0)  # Reverse y-axis to match image coordinates

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Keypoints on Image')

# Show the plot
plt.show()