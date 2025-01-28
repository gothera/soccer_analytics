import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import torchvision.transforms as T
import itertools

class ReIDDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        """
        Args:
            json_path: Path to annotation JSON file
            img_dir: Directory containing images
            transform: Optional transform to be applied on images
        """
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform or T.Compose([
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Group images by frame number and by target/non-target
        self.frames = {}
        self.target_images = []  # All target images across all frames
        self.non_target_images = []  # All non-target images across all frames
        
        for ann in self.annotations:
            frame_num = ann['frame_number']
            if frame_num not in self.frames:
                self.frames[frame_num] = {'target': [], 'others': []}
            
            if ann['is_target']:
                self.frames[frame_num]['target'].append(ann['image'])
                self.target_images.append(ann['image'])
            else:
                self.frames[frame_num]['others'].append(ann['image'])
                self.non_target_images.append(ann['image'])
        
        # Generate all possible valid triplets
        self.triplets = self._generate_all_triplets()
    
    def _generate_all_triplets(self):
        """Generate all possible valid triplets (anchor, positive, negative)"""
        triplets = []
        
        # Generate all possible anchor-positive pairs (excluding same image)
        anchor_positive_pairs = list(itertools.combinations(self.target_images, 2))
        
        # For each anchor-positive pair, use all possible negatives
        for anchor, positive in anchor_positive_pairs:
            # Use as anchor-positive and positive-anchor
            for neg in self.non_target_images:
                triplets.append((anchor, positive, neg))
                triplets.append((positive, anchor, neg))
        
        print(f"Generated {len(triplets)} triplets")
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        
        anchor = Image.open(os.path.join(self.img_dir, anchor_path)).convert('RGB')
        positive = Image.open(os.path.join(self.img_dir, pos_path)).convert('RGB')
        negative = Image.open(os.path.join(self.img_dir, neg_path)).convert('RGB')
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
        }
if __name__ == "__main__":
    data_dir='annotations_20241204_224523/annotations.json'
    json_path='annotations_20241204_224523/annotations.json'
    # Create dataset instance
    dataset = ReIDDataset(
        json_path,
        data_dir
    )
    print(len(dataset))
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)