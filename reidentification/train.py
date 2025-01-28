import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from keypoints.dataset import ReIDDataset

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class LightReIDModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(LightReIDModel, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 32)
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def train_reid(
    dataset_path,
    img_dir,
    num_epochs=50,
    batch_size=32,
    embedding_dim=128,
    learning_rate=0.001,
    margin=0.3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Initialize dataset and dataloader
    dataset = ReIDDataset(dataset_path, img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model, loss, and optimizer
    model = LightReIDModel(embedding_dim=embedding_dim).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        for batch_idx, xs in enumerate(dataloader):
            # Move data to device
            (anchor, positive, negative) = xs['anchor'], xs['positive'], xs['negative']
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            # Calculate loss
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Update statistics
            epoch_loss += loss.item()
            valid_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / valid_batches
        print(f'Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_epoch_loss:.4f}')
        
        # Update learning rate
        scheduler.step(avg_epoch_loss)
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_reid_model.pth')
    
    return model

# Function to extract embeddings for inference
def extract_embedding(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        embedding = model(image_tensor.unsqueeze(0).to(device))
    return embedding.cpu().numpy()

# Usage example
if __name__ == "__main__":
    # Training parameters
    params = {
        'dataset_path': 'annotations_20241204_224523/annotations.json',
        'img_dir': 'annotations_20241204_224523',
        'num_epochs': 50,
        'batch_size': 32,
        'embedding_dim': 128,
        'learning_rate': 0.001,
        'margin': 0.3
    }
    
    # Train model
    model = train_reid(**params)
    
    # Save final model
    torch.save(model.state_dict(), 'final_reid_model.pth')