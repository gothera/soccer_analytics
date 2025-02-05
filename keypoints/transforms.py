import random
from typing import Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from datatools.reader import decode_annot
from datatools.ellipse_utils import PERP_LINES, POINTS_LEFT, POINTS_RIGHT
from datatools.intersections import get_intersections


class ColorAugment:
    def __init__(self,
                 brightness: Tuple[float, float] = (0.8, 1.2),
                 color: Tuple[float, float] = (0.8, 1.2),
                 contrast: Tuple[float, float] = (0.8, 1.2)):
        self.brightness = brightness
        self.color = color
        self.contrast = contrast

    def _img_aug(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(float)
        random_colors = np.random.uniform(
            self.brightness[0], self.brightness[1])\
            * np.random.uniform(self.color[0], self.color[1], 3)
        for i in range(3):
            img[:, :, i] = img[:, :, i] * random_colors[i]
        mean = img.mean(axis=(0, 1))
        contrast = np.random.uniform(self.contrast[0], self.contrast[1])
        img = (img - mean) * contrast + mean
        img = np.clip(img, 0.0, 255.0)
        img = img.astype(np.uint8)
        return img

    def __call__(self, sample):
        sample['image'] = self._img_aug(sample['image'])
        return sample


class GaussNoise:
    def __init__(self, sigma_sq: float = 30.0):
        self.sigma_sq = sigma_sq

    def __call__(self, sample):
        img = sample['image'].astype(int)
        w, h, c = img.shape
        gauss = np.random.normal(0, np.random.uniform(0.0, self.sigma_sq),
                                 (w, h, c))
        img = img + gauss
        img = np.clip(img, 0, 255)
        sample['image'] = img.astype(np.uint8)
        return sample


class ToTensor:
    """Transform opencv image to tensor. Should be the last transform.
    """

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        sample['image'] = self.to_tensor(sample['image'])
        return sample


class UseWithProb:
    """Apply transform with a given probability for data augmentation.

    Args:
        transform (Callable): Transform to apply.
        prob (float, optional): Probability of the transform. Should be in
            range [0..1]. Defaults to 0.5.
    """

    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample = self.transform(sample)
        return sample


FLIP_POSTS = {
    'Goal left post right': 'Goal left post left ',
    'Goal left post left ': 'Goal left post right',
    'Goal right post right': 'Goal right post left',
    'Goal right post left': 'Goal right post right'
}


def swap_top_bottom_names(line_name: str) -> str:
    x: str = 'top'
    y: str = 'bottom'
    if x in line_name or y in line_name:
        return y.join(part.replace(y, x) for part in line_name.split(x))
    return line_name


def swap_posts_names(line_name: str) -> str:
    if line_name in FLIP_POSTS:
        return FLIP_POSTS[line_name]
    return line_name


def flip_annot_names(annot, swap_top_bottom: bool = True,
                     swap_posts: bool = True):
    annot = mirror_labels(annot)
    if swap_top_bottom:
        annot = {swap_top_bottom_names(k): v for k, v in annot.items()}
    if swap_posts:
        annot = {swap_posts_names(k): v for k, v in annot.items()}
    return annot


class Flip:
    """Horizontal image flip.
    """

    def __call__(self, sample):
        sample['image'] = cv2.flip(sample['image'], 1)
        sample['annot'] = flip_annot_names(sample['annot'])
        for line in sample['annot']:
            for point in sample['annot'][line]:
                point['x'] = 1.0 - point['x']

        return sample

class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def train_transform(brightness: Tuple[float, float] = (0.8, 1.2),
                    color: Tuple[float, float] = (0.8, 1.2),
                    contrast: Tuple[float, float] = (0.8, 1.2),
                    gauss_noise_sigma: float = 30.0,
                    prob: float = 0.5):
    transforms = ComposeTransform([
        UseWithProb(ColorAugment(brightness=brightness,
                                 color=color,
                                 contrast=contrast), prob),
        UseWithProb(GaussNoise(gauss_noise_sigma), prob),
        # UseWithProb(Flip(), 0.5),
        ToTensor()
    ])
    return transforms


def test_transform():
    transforms = ComposeTransform([
        ToTensor()
    ])
    return transforms


class HRNetPredictionTransform:
    def __init__(self, size):
        self.H, self.W = size

    def __call__(self, preds):
        _, _, h, w = preds.shape
        preds = torch.exp(preds)
        x_prob, x = torch.max(torch.max(preds, dim=2)[0], dim=2)
        y_prob, y = torch.max(torch.max(preds, dim=3)[0], dim=2)
        conf = torch.min(x_prob, y_prob)
        x = x*2
        y = y*2

        # (B, N, 3)
        predictions = torch.stack([x, y, conf], dim=-1)[:, :-1, :]
        return predictions
