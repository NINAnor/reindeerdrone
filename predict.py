import os
import cv2
import torch
import numpy as np
import yaml
from yaml import FullLoader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import sys

from utils import tile_image


def load_predictor(config_file, model_weights, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class (reindeer)
    predictor = DefaultPredictor(cfg)
    return predictor


def predict(image, predictor, tile_size, overlap):
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
    return results


def visualize_preds(image, results, output_path):
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
    model_weights = cfg["MODEL_WEIGHTS"]
    output_path = cfg["OUTPUT_FOLDER"]
    tile_size = cfg["TILE_SIZE"]
    overlap = cfg["OVERLAP"]
    output_path = os.path.join(output_path, os.path.basename(image_path))

    image = cv2.imread(image_path)

    # Load the predictor
    predictor = load_predictor(config_file, model_weights)

    # Perform prediction and visualize the results
    results = predict(image, predictor, tile_size, overlap)
    visualize_preds(image, results, output_path)


if __name__ == "__main__":

    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=FullLoader)
    os.makedirs(cfg["OUTPUT_FOLDER"], exist_ok=True)

    image_path = sys.argv[1]

    main(image_path, cfg)
