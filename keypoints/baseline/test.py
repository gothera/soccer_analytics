import cv2
import numpy as np

# Load images
match_img = cv2.imread('match_image.jpg')
court_template = cv2.imread('court_template.jpg')

# Define corresponding points in both images
# These should be the 4 points you have identified
src_points = np.array([
    [
        405.5,
        697.6
    ],  # Point 1 in template
    [
    1205.8,
    787.0],  # Point 2 in template
    [
        1395.4,
        596.0
    ],  # Point 3 in template
    [
    678.2,
          522.2
        ]   # Point 4 in template
], dtype=np.float32)

dst_points = np.array([
    [X1, Y1],  # Corresponding point 1 in match image
    [X2, Y2],  # Corresponding point 2 in match image
    [X3, Y3],  # Corresponding point 3 in match image
    [X4, Y4]   # Corresponding point 4 in match image
], dtype=np.float32)

# Calculate homography matrix
H, _ = cv2.findHomography(src_points, dst_points)

# Warp template image
warped_template = cv2.warpPerspective(
    court_template, 
    H, 
    (match_img.shape[1], match_img.shape[0])
)

# Create overlay
alpha = 0.5
overlay = cv2.addWeighted(match_img, alpha, warped_template, 1-alpha, 0)

# Display or save results
cv2.imwrite('overlay_result.jpg', overlay)