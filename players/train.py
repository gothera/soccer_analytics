from ultralytics import YOLO
import os
import argparse
from datetime import datetime

def train_yolo(
    data_yaml,
    model_size='n',  # n, s, m, l, x
    epochs=200,
    batch_size=8,
    imgsz=1920,
    project='yolo_soccer',
    resume=False,
    device='',  # auto-select
    patience=50,  # early stopping patience
    save_period=10,  # save checkpoint every X epochs
):
    """
    Train YOLOv8 model on custom dataset
    """
    # Create unique run name based on timestamp
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize model
    model_name = f'yolov8{model_size}.pt'
    model = YOLO(model_name)
    
    # Training arguments
    args = dict(
        data=data_yaml,  # path to data.yaml
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=project,
        name=run_name,
        save_period=save_period,
        save=True,  # save checkpoints
        plots=True,  # save plots and visualizations
        exist_ok=True,  # overwrite existing experiment
        pretrained=True,  # use pretrained model
        resume=resume,  # resume training from last checkpoint
        verbose=True,  # print verbose output
        
        # Hyperparameters
        lr0=0.01,     # initial learning rate
        lrf=0.01,     # final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,      # box loss gain
        cls=0.5,      # cls loss gain
        dfl=1.5,      # dfl loss gain
        
        # Other parameters
        workers=8,    # number of worker threads
        seed=0,       # random seed
        deterministic=True,  # deterministic training
    )
    
    # Start training
    try:
        results = model.train(**args)
        
        # Print training summary
        print("\nTraining completed successfully!")
        print(f"Results saved to {os.path.join(project, run_name)}")
        
        # Get best model path
        best_model = os.path.join(project, run_name, 'weights', 'best.pt')
        
        if os.path.exists(best_model):
            print(f"Best model saved to: {best_model}")
            
            # Validate best model
            print("\nValidating best model...")
            metrics = model.val()
            
            print("\nValidation Metrics:")
            print(f"mAP50: {metrics.box.map50:.3f}")
            print(f"mAP50-95: {metrics.box.map:.3f}")
            
        return best_model
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 on custom dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1920, help='Image size')
    parser.add_argument('--project', type=str, default='yolo_soccer', help='Project name')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (empty for auto)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every X epochs')
    
    args = parser.parse_args()
    
    # Train model
    best_model = train_yolo(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        project=args.project,
        resume=args.resume,
        device=args.device,
        patience=args.patience,
        save_period=args.save_period
    )