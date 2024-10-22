import os
import json
import concurrent.futures as cf

import cv2
import torch
from tqdm import tqdm
from argus import load_model
from torchvision import transforms as T

from datatools.ellipse_utils import PITCH_POINTS
from metamodel import HRNetMetaModel
from prediction import CameraCreator

DEVICE = 'cpu'  # Device for running inference
# Update the model path to include the best actual keypoints model
MODEL_PATH = './network.pth'
IMG_DIR = './dataset/valid'
SAVE_DIR = './challenge_003'
# Lines model prediction file or None (lines are ignored in that case)
LINES_FILE = None


batch_size = 8
NUM_WORKERS = 16


class PredictionSaver:
    def __init__(self, pred2cam):
        self.pred2cam = pred2cam

    def __call__(self, x) -> int:
        pred, json_path, img_name = x
        cam = self.pred2cam(pred, img_name)
        if cam is not None:
            with open(json_path, "w") as f:
                json.dump(cam.to_json_parameters(), f, indent=4)
            return 1
        return 0


def main():
    to_tensor = T.ToTensor()
    os.makedirs(SAVE_DIR, exist_ok=True)
    calibrator = CameraCreator(
        PITCH_POINTS, conf_thresh=0.5, conf_threshs=[0.5, 0.35, 0.2],
        algorithm='iterative_voter',
        lines_file=LINES_FILE, max_rmse=55.0, max_rmse_rel=5.0,
        min_points=5, min_focal_length=10.0, min_points_per_plane=6,
        min_points_for_refinement=6, reliable_thresh=57)
    model = load_model(MODEL_PATH, loss=None, optimizer=None, device=DEVICE)

    executor = cf.ProcessPoolExecutor(max_workers=NUM_WORKERS)
    worker = PredictionSaver(calibrator)
    pred_imgs = 0
    img_names = [img for img in os.listdir(IMG_DIR) if img.endswith('.png')]
    total_imgs = len(img_names)

    for i in tqdm(range(0, total_imgs, batch_size)):
        batch_imgs = img_names[i:i+batch_size]
        img_paths = [os.path.join(IMG_DIR, img_name)
                     for img_name in batch_imgs]
        json_paths = [os.path.join(
            SAVE_DIR, 'camera_'+img_name.replace('.png', '.json'))
            for img_name in batch_imgs]
        tensors = [to_tensor(cv2.imread(img_path)) for img_path in img_paths]
        tensor = torch.stack(tensors, dim=0)
        preds = model.predict(tensor).cpu().numpy()
        for res in executor.map(worker, [(preds[i], json_paths[i],
                                          batch_imgs[i])
                                         for i in range(preds.shape[0])]):
            if res is not None:
                pred_imgs += res

    print(f'Completeness: {pred_imgs/total_imgs:.2f}')


if __name__ == '__main__':
    main()
