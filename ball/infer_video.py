from model import BallTrackerNet
import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

import threading
from queue import Queue
from typing import List, Tuple
from tqdm import tqdm
from itertools import groupby
from scipy.spatial import distance


height = 720
width = 1280
num_workers = 1
batch_size = 2
device = 'cuda'
global_results = []

# Function to process a batch of frame sequences
def process_batch(model, batch, start_index):
    with torch.no_grad():
        outputs = model(batch.float().to(device))
        out = outputs.argmax(dim=1).detach().cpu().numpy()
        # print(out.shape)
        for i, output in enumerate(out):
            x_pred, y_pred = postprocess(output)
            global_results[start_index + i] = (x_pred, y_pred)
        
# Worker thread function
def worker(model, input_queue):
    while True:
        task = input_queue.get()
        if task == None:
            break
        batch, start_index = task
        process_batch(model, batch, start_index)
        input_queue.task_done()
        # print("Finished task", start_index)


def display_bw_image(tensor_image):
    # Ensure the tensor is on CPU and detached from the computation graph
    tensor_image = tensor_image.cpu().detach()

    # Reshape the tensor to (360, 640)
    image = tensor_image.reshape(height, width)
    
    # Normalize the image to 0-255 range
    image = ((image - image.min()) / (image.max() - image.min()) * 255)
    
    # Convert to numpy array and cast to uint8
    image_np = image.numpy().astype(np.uint8)
    
    # Display the image
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    plt.show()

def postprocess(feature_map, scale=1):
    feature_map *= 255
    feature_map = feature_map.reshape((height, width))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
    x,y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scale
            y = circles[0][0][1]*scale
    return x, y

def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def infer_model(frames, model):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """

    input_queue = Queue()
        
    global global_results    
    global_results = [(None, None)] * (len(frames))
    dists = [-1]*2
    # ball_track = [(None,None)]*2
    batch = []
    for num in range(2, len(frames)):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)
        batch.append(imgs)
        if len(batch) == batch_size:
            np_batch = np.array(batch)
            input_queue.put((torch.from_numpy(np_batch), num - len(batch) + 1))
            batch = []
        # out = model(torch.from_numpy(inp).float().to(device))
        # # display_bw_image(out.argmax(dim=1)[0])
        # output = out.argmax(dim=1).detach().cpu().numpy()
        # x_pred, y_pred = postprocess(output)
        # ball_track.append((x_pred, y_pred))
    
    # Signal worker threads to exit
    # for _ in range(num_workers):
    #     input_queue.put(None)
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, args=(model, input_queue))
        t.start()
        threads.append(t)

    for _ in range(num_workers):
        input_queue.put(None)
    # Wait for all tasks to complete
    # input_queue.join()

    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    ball_track = global_results
    for i in range(2, len(ball_track)):
        if ball_track[i-1][0] and ball_track[i-2][0]:
            dist = distance.euclidean(ball_track[i-1], ball_track[i-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists

def remove_outliers(ball_track, dists, max_dist = 100):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=3, max_dist_gap=120, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track

def write_track(frames, ball_track, path_output_video, fps, trace=3):
    """ Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """
    height, width = frames[0].shape[:2]
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_output_video, fourcc, fps, (width, height))
    cap_output = cv2.VideoCapture('./cut_1.mp4')

    for num in range(len(frames)):
        ret_output, output_frame = cap_output.read()
        if not ret_output:
            print("failed writing frame at", num)
            continue
        ball_point = ball_track[num]
       
        # for i in range(trace):
        #     if (num-i > 0):
        #         if ball_track[num-i][0]:
        #             x = int(ball_track[num-i][0])
        #             y = int(ball_track[num-i][1])
        #             output_frame = cv2.circle(output_frame, (x,y), radius=1, color=(0, 0, 255), thickness=3-i)
        #         else:
        #             break
        if ball_track[num][0]:
            x = int(ball_track[num][0])
            y = int(ball_track[num][1])
            output_frame = cv2.circle(output_frame, (x,y), radius=5, color=(0, 0, 255), thickness=6)
        out.write(output_frame) 
    out.release()
    cap_output.release()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_path', type=str, default='./model_last_nba_ball.pt', help='path to model')
    parser.add_argument('--video_path', type=str, default='cut_1.mp4')
    parser.add_argument('--video_out_path', type=str, default='./output_points_ball.mp4')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track extrapolation')
    args = parser.parse_args()
    
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    frames, fps = read_video(args.video_path)
    ball_track, dists = infer_model(frames, model)
    ball_track = remove_outliers(ball_track, dists)
    
    if args.extrapolation:
        subtracks = split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
        
    write_track(frames, ball_track, args.video_out_path, fps)
    print("global", ball_track)
    
    
    
    
    