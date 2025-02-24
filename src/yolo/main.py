import logging
import hydra
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    logging.info(f"Application is starting with configuration: {cfg}")
    logging.info(f"Data directory is set to: {cfg.paths.ROOT_DATASET_PATH}")
    
    root_data_dir = Path(cfg.paths.ROOT_DATASET_PATH)
    
    data_yaml = root_data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data file not found: {data_yaml}")
    
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    model.train(
        project="training_logs",
        data=data_yaml,
        epochs=cfg.train.EPOCHS,
        imgsz=cfg.train.IMG_SIZE,
        workers=cfg.train.NUM_WORKERS,
        batch=cfg.train.BATCH_SIZE,
    )

if __name__ == "__main__":
    main()
