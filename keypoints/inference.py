import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import hydra
import time

from omegaconf import DictConfig
from argus import load_model
from keypoints.transforms import test_transform, HRNetPredictionTransform
from argus.utils import deep_to, deep_detach
from datatools.ellipse_utils import INTERSECTON_TO_PITCH_POINTS, PITCH_POINTS, get_homography
from datatools.geom import point_within_img

CONFIG_PATH='./val_config.yaml'

def plot_image_points(img, pred):
    # print(pred, "pred", pred.shape)
    conf = pred[:, 2]
    xs, ys = [None] * 57 , [None] * 57
    # print(conf, "da")
    world_points, ground_points = [], []
    for i, c in enumerate(conf):
        if c > 0.5:
            xs[i] = float(pred[i, 0])
            ys[i] = float(pred[i, 1])
            print(i, pred[i, 0], pred[i, 1])
            world_points.append((xs[i], ys[i]))
            ground_points.append(PITCH_POINTS[INTERSECTON_TO_PITCH_POINTS[i]])
    H = get_homography(np.array(ground_points), np.array(world_points))
    for idx in INTERSECTON_TO_PITCH_POINTS.keys():
        if idx > 29 and pred[idx][2] < 0.5:
            world_p = PITCH_POINTS[INTERSECTON_TO_PITCH_POINTS[idx]]
            img_p = H @ np.array([world_p[0], world_p[1], 1])
            coords = point_within_img(img_p[:2] / img_p[2])
            if coords is None:
                continue
            xs[idx]= coords[0]
            ys[idx]= coords[1]
    # print(points)
    # plot_lines(xs, ys)
    plt.plot(xs, ys, 'o')
    plt.imshow(img)
    plt.show()

def plot_lines(xs, ys):
    lines = {
        4:5,
        9:8,
        17:16,
        11:9,
        10:8,
        5:7,
        6:4,
        19:17,
        18:16,
        14:15,
        21:20,
        23:21,
        22:20,
        29:23
    }

    for k, v in lines.items():
        if xs[k] is None or xs[v] is None:
            continue
        x = xs[k], xs[v]
        y = ys[k], ys[v]
        plt.plot(x, y, color='b')


@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH), config_name='val_config')
def main(cfg: DictConfig):
    # Open the input video
    model = hydra.utils.instantiate(cfg.model)
    pretrain_path = cfg.model.params.pretrain
    print("path", pretrain_path)
    if os.path.exists(pretrain_path):
        model = load_model(pretrain_path,
                           device=cfg.model.params.device)
    if model is None:
        print('[ERROR] Failed to load pre-trained model')
        return
    input_video_path = "./game_cuts/cut_1.mp4"
    output_video_path = "output_video_1.mp4"

    cap = cv2.VideoCapture(input_video_path)
    tfms = test_transform()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    prediction_transform = HRNetPredictionTransform((270,480))
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        # Normalize pixel values to [0, 1]
        input_frame = np.rollaxis(frame, 2, 0)
        sample = {'image': frame}
        # Convert to PyTorch tensor
        input_tensor = tfms(sample)
        input_tensor['image'] = input_tensor['image'].unsqueeze(0)
        print(frame.shape, input_tensor['image'].shape)
        prediction = model.nn_module(input_tensor['image'])
        prediction = deep_detach(prediction)
        points = prediction_transform(prediction[-1])
        # print(points)
        plot_image_points(input_tensor['image'][0].permute(1,2,0).detach().cpu().numpy(), points[0].detach().cpu().numpy())
            
    # Release video objects
    cap.release()
    out.release()

if __name__ == "__main__":
    main()
