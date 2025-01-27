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
    plt.imshow(image_tensor.permute(1,2,0))
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

def plot_heatmap(heatmap_tensor):
    """
    Plots a heatmap of a tensor with shape (68, 120) using matplotlib.

    Parameters:
    heatmap_tensor (numpy.ndarray): The heatmap tensor to plot. It should have shape (68, 120).
    """
    heatmap = heatmap_tensor.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (960, 540))
    plt.imshow(heatmap, cmap='viridis', aspect='auto')
    plt.show()

def custom_collate(batch):
    default_collated = default_collate([{k: v for k, v in sample.items()
                                         if k in collate_objs}
                                        for sample in batch])
    custom_collated = {'img_name': [sample['img_name'] for sample in batch]}

    return {**default_collated, **custom_collated}

def plot_img_keypoints(img, heatmap):
    # Assuming 'image' is your input image of shape (540, 960, 3)
    # and 'heatmap' is your black-and-white heatmap of shape (1, 270, 480)

    # Step 1: Remove the first dimension from the heatmap
    heatmap = heatmap.squeeze().detach().cpu().numpy()

    # Step 2: Resize the heatmap to match the size of the image (540, 960)
    heatmap_resized = cv2.resize(heatmap, (960, 540))

    # Step 3: Normalize the heatmap to range between 0 and 255
    heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)

    # Step 4: Convert heatmap to a 3-channel image (Grayscale to RGB)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    img_transposed = img.astype(np.float32)
    heatmap_colored = heatmap_colored.astype(np.float32)
    print(f"Image shape: {img.shape}")
    print(f"Heatmap shape: {heatmap_colored.shape}")
    # print(img)
    # Step 5: Overlay the heatmap on the original image
    overlay = cv2.addWeighted(img_transposed, 0.5, heatmap_colored, 0.5, 0)
    overlay = overlay.astype(np.uint8)

    # Step 6: Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct color display
    plt.axis('off')
    plt.show()


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
        for img_path in sorted(os.listdir(os.path.join(dataset_folder, 'images'))):
            self._img_paths.append(os.path.join(dataset_folder, 'images', img_path))
            self._img_names.append(img_path)
        for annot_path in sorted(os.listdir(os.path.join(dataset_folder, 'json_labels'))):
            self._annot_paths.append(os.path.join(dataset_folder, 'json_labels', annot_path))

    def __getitem__(self, idx):
        image = cv2.imread(self._img_paths[idx], cv2.IMREAD_COLOR)
        sample = {'image': image}
        if self._transform:
            sample = self._transform(sample)
        # print(self._img_paths[:3])
        # print(self._annot_paths)
        keypoints, mask = self._annot2keypoints(self._annot_paths[idx])
        sample['keypoints'] = keypoints
        sample['img_idx'] = idx
        sample['mask'] = mask
        sample['img_name'] = self._img_names[idx]
        print(self._img_names[idx])
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
               transform: Optional[Callable] = None, shuffle: bool = True)\
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
    dataset = HRNetDataset(dataset_folder='/Users/cosmincojocaru/playground/keypoints/keypoints_dataset/hrnet_keypoints_dataset/train')
    sample = dataset[0]
    # plt.imshow(img)
    # plt.show()
    train_loader = get_loader(cfg.data.train, cfg.data_params, None, True)
    dl = iter(train_loader)

    # batch = next(dl)
    # img, keypoints, mask = batch['image'][0], batch['keypoints'][0].reshape(-1, cfg.data_params.num_keypoints, 3), batch['mask'][0]
    # print(img.shape, keypoints.shape, mask.shape)

    for batch in dl:
        for idx in range(cfg.data_params.batch_size):
            img, keypoints, mask = batch['image'][idx], batch['keypoints'][idx].reshape(-1, cfg.data_params.num_keypoints, 3), batch['mask'][idx]
            # print(img.shape, keypoints.shape, mask.shape)
            heatmaps = create_heatmaps(keypoints, 1.0)
            heatmaps = torch.cat(
                    [heatmaps, (1.0 - torch.max(heatmaps, dim=1, keepdim=True)[0])], 1)
            maps = torch.sum(heatmaps[0][:-1], 0)
            # plot_heatmap(maps)

            plot_img_keypoints(img.detach().cpu().numpy(), maps)



if __name__ == "__main__":
    main()