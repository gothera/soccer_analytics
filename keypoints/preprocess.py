import cv2
import json
import keypoints.constants as constants
import matplotlib.pyplot as plt
from PIL import Image
from key_point_processing import *

def load_img_with_data(id):
    img_name = '0' * (5 - len(str(id))) + str(id)
    image_path = f"./dataset/valid/valid/{img_name}.jpg"
    img = cv2.imread(image_path)
    pil_img = Image.open(image_path)

    with open(f"./dataset/valid/valid/{img_name}.json") as file:
        img_data = json.load(file)

    return img, pil_img, img_data

def preprocess_img(id):
    img, pil_img, data = load_img_with_data(id)
    height, width, _ = img.shape
    for k, v in data.items():
        if k.find("Circle") != -1:
            # for point in v:
            #     p = get_point_coordinates(point)
            #     cv2.circle(img, p, radius=2, color=(0, 0, 255), thickness=-1)
            find_ellipse(pil_img, v)
            # for point in fitted_points:
            #     print(point)
            #     cv2.circle(img, point, radius=2, color=(255, 0, 0), thickness=-1)
            # continue


        slope, intercept = fit_line_to_points(v)
        x1 = 0
        y1 = int(slope * x1 + intercept)
        x2 = constants.IMG_WIDTH - 1
        y2 = int(slope * x2 + intercept)

    plt.imshow(img)
    plt.show()      
    return img, data

if __name__ == "__main__":
    id = 67
    img, img_data = preprocess_img(id)
    # top_right, top_left, bottom_right, bottom_left = get_rectangle("Small", "right", img_data)
    points = get_area("Small", "right", img_data)
    
    middle_line_points = get_middle_points(img_data)
    points += middle_line_points
    # points = get_pitch_corners(img_data)
    for p in points:
        if p == ():
            continue
        cv2.circle(img, p, radius=2, color=(0, 255, 255), thickness=3)

    plt.imshow(img)
    plt.show() 