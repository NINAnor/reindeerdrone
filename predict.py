import os
import cv2
import torch
import numpy as np
import yaml
import json
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
                results.append((box, score, x, y))
    return results


def visualize_preds(image, results, output_path):
    # Visualize predictions on the original image
    for box, score, x_offset, y_offset in results:
        x1, y1, x2, y2 = box
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")


def save_predictions_to_json(results, output_json_path):
    detections = []
    for box, score, x_offset, y_offset in results:
        x1, y1, x2, y2 = box
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
            }
        )
    with open(output_json_path, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"Predictions saved to {output_json_path}")
    print(f"{len(detections)} detections have been made")


def main(image_path, cfg):

    # Folder management
    os.makedirs(os.path.join(cfg["OUTPUT_FOLDER"], "image"), exist_ok=True)
    os.makedirs(os.path.join(cfg["OUTPUT_FOLDER"], "json"), exist_ok=True)
    output_image_path = os.path.join(
        cfg["OUTPUT_FOLDER"],
        "image",
        f"{os.path.splitext(os.path.basename(image_path))[0]}_pred.png",
    )
    output_json_path = os.path.join(
        cfg["OUTPUT_FOLDER"],
        "json",
        f"{os.path.splitext(os.path.basename(image_path))[0]}_pred.json",
    )

    image = cv2.imread(image_path)

    # Load the predictor
    predictor = load_predictor(cfg["CONFIG_FILE"], cfg["MODEL_WEIGHTS"])

    # Perform prediction and visualize the results
    results = predict(image, predictor, cfg["TILE_SIZE"], cfg["OVERLAP"])

    if cfg["PLOT_PREDICTION"]:
        visualize_preds(image, results, output_image_path)

    save_predictions_to_json(results, output_json_path)


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=FullLoader)

    image_path = sys.argv[1]

    main(image_path, cfg)
