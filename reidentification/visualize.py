import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from matplotlib.widgets import Button
from keypoints.dataset import *

class TripletVisualizer:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        
        # Create denormalization transform
        self.denorm = transforms.Compose([
            transforms.Normalize(
                mean=[0, 0, 0],
                std=[1/0.229, 1/0.224, 1/0.225]
            ),
            transforms.Normalize(
                mean=[-0.485, -0.456, -0.406],
                std=[1, 1, 1]
            )
        ])
        
        # Create the figure and axes
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.canvas.manager.set_window_title('Triplet Viewer')
        
        # Initialize the plot
        self.show_next_triplet()
        
        # Connect key event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add instruction text
        self.fig.text(0.5, 0.02, 
                     'Press any key to show next triplet\nPress Q to quit',
                     ha='center', va='center')
        
        plt.show()
    
    def process_image(self, img):
        """Denormalize and convert tensor to numpy array"""
        img = self.denorm(img[0]).permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0, 1)
    
    def show_next_triplet(self):
        """Display the next triplet"""
        try:
            batch = next(self.iterator)
            
            # Get images and frame number
            anchor = self.process_image(batch['anchor'])
            positive = self.process_image(batch['positive'])
            negative = self.process_image(batch['negative'])
            frame_num = batch['frame_num'][0].item()
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Show images
            self.axes[0].imshow(anchor)
            self.axes[0].set_title(f'Anchor\nFrame {frame_num}')
            self.axes[0].axis('off')
            
            self.axes[1].imshow(positive)
            self.axes[1].set_title(f'Positive\nFrame {frame_num}')
            self.axes[1].axis('off')
            
            self.axes[2].imshow(negative)
            self.axes[2].set_title(f'Negative\nFrame {frame_num}')
            self.axes[2].axis('off')
            
            plt.tight_layout()
            self.fig.canvas.draw()
            
            return True
            
        except StopIteration:
            print("\nEnd of dataset reached!")
            plt.close()
            return False
    
    def on_key_press(self, event):
        """Handle key press events"""
        if event.key.lower() == 'q':
            plt.close()
        else:
            self.show_next_triplet()

def visualize_dataset_interactive(data_dir, json_path):
    """
    Create an interactive visualization of the dataset
    
    Args:
        data_dir: Path to the directory containing images
        json_path: Path to the JSON annotations file
    """
    # Create dataset instance
    dataset = ReIDDataset(
        json_path,
        data_dir
    )

    # Create DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create visualizer for training set
    TripletVisualizer(dataloader)
    
    # Ask if user wants to see validation set
    # response = input("\nDo you want to see validation samples? (y/n): ")
    # if response.lower() == 'y':
    #     print("\nShowing validation samples...")
    #     print("Press any key to advance to next triplet")
    #     print("Press Q to quit")
    #     TripletVisualizer(val_loader)

# Example usage
if __name__ == '__main__':
    data_dir='annotations_20241204_224523'
    json_path='annotations_20241204_224523/annotations.json' 
    visualize_dataset_interactive(
        data_dir=data_dir,
        json_path=json_path
    )