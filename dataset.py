import os
from typing import List, Tuple, Callable, Optional

import cv2
import numpy as np
import hydra

from omegaconf import DictConfig
from torch.utils.data import DataLoader, ConcatDataset, default_collate
from torch.utils.data.dataset import Dataset

from datatools.reader import read_json, decode_annot
from datatools.intersections import get_intersections
from baseline.points import scale_points

CONFIG_PATH = './train_config.yaml'

collate_objs = ['keypoints', 'image', 'img_idx', 'mask']

import matplotlib.pyplot as plt
from loss import *

def plot_image_tensor(image_tensor):
    """
    Plots an image tensor of shape (540, 960, 3) using matplotlib.

    Parameters:
    image_tensor (numpy.ndarray): The image tensor to plot. It should have shape (540, 960, 3).

    """
    plt.imshow(image_tensor)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

def plot_heatmap(heatmap_tensor):
    """
    Plots a heatmap of a tensor with shape (68, 120) using matplotlib.

    Parameters:
    heatmap_tensor (numpy.ndarray): The heatmap tensor to plot. It should have shape (68, 120).
    """
    plt.imshow(heatmap_tensor, cmap='viridis', aspect='auto')
    plt.show()

def custom_collate(batch):
    default_collated = default_collate([{k: v for k, v in sample.items()
                                         if k in collate_objs}
                                        for sample in batch])
    custom_collated = {'img_name': [sample['img_name'] for sample in batch]}

    return {**default_collated, **custom_collated}


class HRNetDataset(Dataset):
    def __init__(self, dataset_folder: str, transform: Optional[Callable] = None,
                 num_keypoints: int = 30,
                 img_size: Tuple[int, int] = (960, 540),
                 margin: float = 0.0):
        super().__init__()
        self._dataset_folder = dataset_folder
        self.num_keypoints = num_keypoints
        self._transform = transform
        self.img_size = img_size
        self.margin = margin
        self._img_paths = []
        self._img_names = []
        self._annot_paths = []
        no = 0
        for fname in sorted(os.listdir(dataset_folder)):
            if 'info' not in fname:
                annot_path = os.path.join(dataset_folder, fname)
                print(annot_path)
                if annot_path.endswith('.json'):
                    img_path = annot_path.replace('.json', '.png')
                    if os.path.exists(img_path):
                        self._img_names.append(fname.replace('.json', '.png'))
                        self._img_paths.append(img_path)
                        self._annot_paths.append(annot_path)
                        no += 1
                        if no >= 2:
                            break

    def __getitem__(self, idx):
        image = cv2.imread(self._img_paths[idx], cv2.IMREAD_COLOR)
        sample = {'image': image}
        if self._transform:
            sample = self._transform(sample)
        keypoints, mask = self._annot2keypoints(self._annot_paths[idx])
        sample['keypoints'] = keypoints
        sample['img_idx'] = idx
        sample['mask'] = mask
        sample['img_name'] = self._img_names[idx]
        return sample

    def _annot2keypoints(self, annot) -> np.ndarray:
        kpts_dict, mask = decode_annot(annot, self.num_keypoints)
        keypoints = np.ones(self.num_keypoints * 3, dtype=np.float32) * -1
        for i in range(self.num_keypoints):
            if kpts_dict[i] is not None:
                keypoints[i * 3] = kpts_dict[i][0]
                keypoints[i * 3 + 1] = kpts_dict[i][1]
                keypoints[i * 3 + 2] = 1
            else:
                keypoints[i * 3 + 2] = 0
        mask_vector = np.ones(self.num_keypoints+1, dtype=int)
        for i in mask:
            mask_vector[i] = 0
        return keypoints, mask_vector

    def __len__(self):
        return len(self._img_paths)


def get_loader(dataset_paths: List[str], data_params: DictConfig,
               transform: Optional[Callable] = None, shuffle: bool = False)\
        -> DataLoader:
    datasets = []
    for dataset_path in dataset_paths:
        datasets.append(HRNetDataset(dataset_path, transform=transform,
                                     num_keypoints=data_params.num_keypoints,
                                     margin=data_params.margin))
    dataset = ConcatDataset(datasets)
    factor = 1 if shuffle else 2
    loader = DataLoader(
        dataset, batch_size=data_params.batch_size * factor,
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        shuffle=shuffle,
        collate_fn=custom_collate)
    return loader

@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH),
            config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def main(cfg: DictConfig):
    dataset = HRNetDataset(dataset_folder='./dataset/train')
    sample = dataset[0]
    # plt.imshow(img)
    # plt.show()
    train_loader = get_loader(cfg.data.train, cfg.data_params, None, True)
    dl = iter(train_loader)

    batch = next(dl)
    img, keypoints, mask = batch['image'][0], batch['keypoints'][0].reshape(-1, cfg.data_params.num_keypoints, 3), batch['mask'][0]
    print(img.shape, keypoints.shape, mask.shape)

    heatmap = create_heatmaps(keypoints, 1.0)
    print(heatmap[0].shape)
    # plot_image_tensor(img)
    print(keypoints[0][16], "da")
    plot_heatmap(heatmap[0][16])



if __name__ == "__main__":
    main()