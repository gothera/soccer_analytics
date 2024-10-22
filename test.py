from dataset import *
from argus import load_model
from argus.utils import deep_to, deep_detach
from transforms import test_transform, train_transform, HRNetPredictionTransform
import cv2

CONFIG_PATH='./val_config.yaml'

def postprocess(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred

def plot_image_points(img, pred):
    # print(pred, "pred", pred.shape)
    conf = pred[:, 2]
    xs, ys = [], []
    # print(conf, "da")
    for i, c in enumerate(conf):
        if c > 0.5:
            xs.append(float(pred[i, 0]))
            ys.append(float(pred[i, 1]))
            print(i, pred[i, 0], pred[i, 1])
    # print(points)
    plt.plot(xs, ys, 'o')
    plt.imshow(img)
    plt.show()

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

    # Step 5: Overlay the heatmap on the original image
    overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)

    # Step 6: Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct color display
    plt.axis('off')
    plt.show()

@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH), config_name='val_config')
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    val_trns = test_transform()
    val_loader = get_loader(cfg.data.val, cfg.data_params, val_trns, True)
    dl = iter(val_loader)
    prediction_transform = HRNetPredictionTransform((270,480))

    batch = next(dl)
    img, keypoints, mask = batch['image'], batch['keypoints'][0].reshape(-1, cfg.data_params.num_keypoints, 3), batch['mask'][0]
    print(img.shape, keypoints.shape, mask.shape)
    plot_image_tensor(img[0])

    heatmaps = create_heatmaps(keypoints, 1.0)
    heatmaps = torch.cat(
            [heatmaps, (1.0 - torch.max(heatmaps, dim=1, keepdim=True)[0])], 1)
    maps = torch.sum(heatmaps[0][:-1], 0)
    # plot_heatmap(maps)

    pretrain_path = cfg.model.params.pretrain
    if os.path.exists(pretrain_path):
        model = load_model(pretrain_path,
                           device=cfg.model.params.device)
    # print(model)
    # plot_heatmap(heatmap[0][16])
    # out = model(img)
    batch = deep_to(batch, device='cpu', non_blocking=True)
    prediction = model.nn_module(batch['image'])
    prediction = deep_detach(prediction)
    points = prediction_transform(prediction[-1])
    # print(points)
    
    # print(prediction[0].shape)
    # pred_masked = prediction[0][0]
    # pred_01 = prediction_transform(torch.exp(pred_masked))  # As we have log on output
    # print(pred_01.shape)
    # plot_heatmap(pred_01[:-1].sum(0).detach())
    # print(pred_01)
    # points = []
    # for kps_num in range(58):
    #     heatmap = (pred_01[kps_num].detach().numpy() * 255).astype(np.uint8)
    #     x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
    #     points.append((x_pred, y_pred))
    #     print(x_pred, y_pred)
    #     ret, heatmap = cv2.threshold(heatmap, 155, 255, cv2.THRESH_BINARY)
    #     plot_img_keypoints(img[0].permute(1,2,0).detach().cpu().numpy(), pred_01[kps_num])
    
    # plot_img_keypoints(img[0].permute(1,2,0).detach().cpu().numpy(), pred_01[:-1].sum(0))
    plot_image_points(img[0].permute(1,2,0).detach().cpu().numpy(), points[0].detach().cpu().numpy())

if __name__ == "__main__":
    main()