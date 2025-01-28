import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from ball.model import BallTrackerNet

def imshow(img, channels=3, title=None):
    img = img.float().to('cpu').numpy()
    img = img.reshape((channels, 720, 1280))
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def display_bw_image(tensor_image):
    # Ensure the tensor is on CPU and detached from the computation graph
    tensor_image = tensor_image.cpu().detach()

    # Reshape the tensor to (720, 1280)
    image = tensor_image.reshape(720, 1280)
    
    # Normalize the image to 0-255 range
    image = ((image - image.min()) / (image.max() - image.min()) * 255)
    
    # Convert to numpy array and cast to uint8
    image_np = image.numpy().astype(np.uint8)
    
    # Display the image
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    plt.show()

def process_video(input_path, output_path, model):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_buffer = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_buffer.append(frame)
        frame_count += 1
        if len(frame_buffer) == 3:
            # Convert frames to the required input shape
            # Convert frames to the required input shape
            input_frames = np.concatenate((frame_buffer[2], frame_buffer[1], frame_buffer[0]), axis=2)
            
            # Normalize pixel values to [0, 1]
            input_frames = input_frames.astype(np.float32) / 255.0
            input_frames = np.rollaxis(input_frames, 2, 0)
            # Transpose to (3, channels, height, width)
        
            # Convert to PyTorch tensor
            input_tensor = torch.from_numpy(input_frames).unsqueeze(0)
            # imshow(input_tensor[0][:3])
            # Add batch dimension
            pred = model(input_tensor)
            output = pred.argmax(dim=1)[0]
            print(output.shape)
            # display_bw_image(output[0])
            # print(output.shape)
             # Reshape heatmap to match frame dimensions (assuming output is (1, 230400))
            heatmap = output.numpy().reshape(720, 1280)
            
            # Normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Create red heatmap
            heatmap_color = np.zeros((720, 1280, 3), dtype=np.uint8)
            heatmap_color[:,:,2] = (heatmap * 255).astype(np.uint8)  # Red channel
            
            # Apply the same heatmap to all three frames
            for frame in frame_buffer:
                overlay = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
                out.write(overlay)
            frame_buffer = []
    
    # Release video objects
    cap.release()
    out.release()

if __name__ == "__main__":
    model_path = './model_last.pt'
    device = 'cpu'
    model = BallTrackerNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    # Usage
    input_video_path = "../game_cuts1/cut_2.mp4"
    output_video_path = "output_video.mp4"

    process_video(input_video_path, output_video_path, model)
