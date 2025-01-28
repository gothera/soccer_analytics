import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import hydra
import time

from omegaconf import DictConfig
from argus import load_model
from transforms import test_transform, HRNetPredictionTransform
from argus.utils import deep_to, deep_detach
from datatools.ellipse_utils import INTERSECTON_TO_PITCH_POINTS, PITCH_POINTS, get_homography
from datatools.geom import point_within_img

CONFIG_PATH='./val_config.yaml'

def plot_image_points(img, pred, out):
    # Define the point sets in order of precedence
    POINT_SETS = [
        {15, 40, 41, 14, 16, 17, 51, 52},  # Set 1
        {15, 40, 41, 14, 16, 17, 51, 52},  # Set 2 (identical to Set 1)
        {15, 40, 41, 14, 16, 17, 52, 19, 21},  # Set 3
        {15, 40, 41, 14, 17, 19, 29},  # Set 5
        {40, 41, 14, 16, 17, 19, 29, 51, 52},
        {40, 41, 15, 16, 17, 18, 19}
    ]
    
    # Initialize arrays for point coordinates
    conf = pred[:, 2]
    xs, ys = [None] * 57, [None] * 57
    
    # Get all visible points from neural network prediction
    visible_points = set()
    for i, c in enumerate(conf):
        if c > 0.5:
            xs[i] = float(pred[i, 0])
            ys[i] = float(pred[i, 1])
            visible_points.add(i)
    
    # # Find the first point set that is completely visible
    # selected_set = None
    # for point_set in POINT_SETS:
    #     if point_set.issubset(visible_points):
    #         selected_set = visible_points
    #         break
    
    # if selected_set is None:
    #     print("No complete point set found in the predictions", visible_points)
    #     selected_set = visible_points
    
    # # Compute homography using only the selected point set
    # world_points, ground_points = [], []
    # for i in selected_set:
    #     world_points.append((xs[i], ys[i]))
    #     ground_points.append(PITCH_POINTS[INTERSECTON_TO_PITCH_POINTS[i]])
    
    # # Compute homography with selected points
    # H = get_homography(np.array(ground_points), np.array(world_points))
    
    # # Project remaining points using computed homography
    # for idx in INTERSECTON_TO_PITCH_POINTS.keys():
    #     if idx > 29 and pred[idx][2] < 0.5:
    #         world_p = PITCH_POINTS[INTERSECTON_TO_PITCH_POINTS[idx]]
    #         img_p = H @ np.array([world_p[0], world_p[1], 1])
    #         coords = point_within_img(img_p[:2] / img_p[2])
    #         if coords is None:
    #             continue
    #         xs[idx] = coords[0]
    #         ys[idx] = coords[1]
    final_xs = [1.3333333 * x if x is not None else None for x in xs]
    final_ys = [1.3333333 * y if y is not None else None for y in ys]
    img = cv2.resize(img, (1280, 720))
    # Create visualization frame
    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Plot points on the image
    plt.plot(final_xs, final_ys, 'o')
    plt.imshow(img)
    
    # Convert matplotlib figure to image
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Write to video
    out.write(frame)
    
    # Clean up matplotlib resources
    plt.close(fig)
    

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
    device = cfg.model.params.device
    print("dev", device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.exists(pretrain_path):
        model = load_model(pretrain_path,
                           device=cfg.model.params.device)
    if model is None:
        print('[ERROR] Failed to load pre-trained model')
        return
    input_video_path = "./subclip_21396.mp4"
    output_video_path = "output_points1.mp4"

    cap = cv2.VideoCapture(input_video_path)
    adj_cap = cv2.VideoCapture('./output_points_ball4.mp4')

    tfms = test_transform()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 720))

    prediction_transform = HRNetPredictionTransform((270,480))
    while True:
        ret, frame = cap.read()
        ret1, out_frame = adj_cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (960, 540))
        # Normalize pixel values to [0, 1]
        sample = {'image': frame}
        # Convert to PyTorch tensor
        input_tensor = tfms(sample)
        input_tensor['image'] = input_tensor['image'].unsqueeze(0)
        # print(frame.shape, input_tensor['image'].shape)
        prediction = model.nn_module(input_tensor['image'].to(device))
        prediction = deep_detach(prediction)
        points = prediction_transform(prediction[-1])
        # print(points)
        plot_image_points(input_tensor['image'][0].permute(1,2,0).detach().cpu().numpy(), points[0].detach().cpu().numpy(), out)
            
    # Release video objects
    cap.release()
    out.release()

if __name__ == "__main__":
    main()
