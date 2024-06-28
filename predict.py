import os
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def load_predictor(config_file, model_weights, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class (reindeer)
    predictor = DefaultPredictor(cfg)
    return predictor

def tile_image(image, tile_size=1024, overlap=100):
    tiles = []
    height, width, _ = image.shape
    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            x1 = min(i + tile_size, width)
            y1 = min(j + tile_size, height)
            tile = image[j:y1, i:x1]
            pad_width = tile_size - (x1 - i)
            pad_height = tile_size - (y1 - j)
            if pad_width > 0 or pad_height > 0:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            tiles.append((tile, i, j))
    return tiles, width, height

def combine_tiles(image, tiles, tile_size=1024, overlap=100):
    height, width, _ = image.shape
    full_image = np.zeros((height, width, 3), dtype=np.uint8)
    for tile, x, y in tiles:
        x1 = min(x + tile_size, width)
        y1 = min(y + tile_size, height)
        pad_width = tile_size - (x1 - x)
        pad_height = tile_size - (y1 - y)
        tile = tile[: tile_size - pad_height, : tile_size - pad_width]
        full_image[y:y1, x:x1] = tile
    return full_image

def predict_and_visualize(predictor, image_path, output_path, tile_size=1024, overlap=100):
    # Read the image
    image = cv2.imread(image_path)
    # Tile the image
    tiles, width, height = tile_image(image, tile_size, overlap)
    results = []
    for tile, x, y in tiles:
        outputs = predictor(tile)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        for box, score in zip(boxes, scores):
            if score >= predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                results.append((box, x, y))
    # Visualize predictions on the original image
    for box, x_offset, y_offset in results:
        x1, y1, x2, y2 = box
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

def main(image_path, cfg):
    config_file = cfg["CONFIG_FILE"]
    model_weights = cfg["MODEL_WEIGHTS"]  # Path to the trained model weights
    output_path = cfg["OUTPUT_FOLDER"]
    output_path = os.path.join(output_path, os.path.basename(image_path))

    # Load the predictor
    predictor = load_predictor(config_file, model_weights)

    # Perform prediction and visualize the results
    predict_and_visualize(predictor, image_path, output_path)


if __name__ == "__main__":

