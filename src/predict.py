import os
import cv2
import yaml
import json
from yaml import FullLoader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm 

from utils import tile_image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_preds_with_gt(image, preds, annotations, output_path, class_names=["Adult", "Calf"]):
    # convert BGR to RGB if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width, _ = image.shape

    # setup figure with image
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image)

    # define colors for classes
    colors = {"adult": "blue", "calf": "green"}
    pred_style = {"linewidth": 1, "facecolor": 'none', "alpha": 0.7}
    gt_style = {"linewidth": 1, "facecolor": 'none', "linestyle": '--', "alpha": 0.7}

    # helper function to draw rectangles
    def draw_rectangle(ax, box, cls, label_text, color, style):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, edgecolor=color, **style
        )
        ax.add_patch(rect)


    # plot predictions
    for box, score, cls, x_offset, y_offset in preds:
        x1, y1, x2, y2 = box
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        
        # determine class color and label
        label = class_names[cls] if cls < len(class_names) else "Unknown"
        color = colors["adult"] if cls == 0 else colors["calf"]
        label_text = f"{label} {score:.2f}"
        
        draw_rectangle(ax, [x1, y1, x2, y2], cls, label_text, color, pred_style)
        
        ax.text(
            x1, y1, label_text, fontsize=6, color='white', 
            weight='bold',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor=color, edgecolor='none', alpha=0.5)
        )

    # plot ground truth
    for annotation in annotations:
        box, cls = annotation
        label = class_names[cls] if cls < len(class_names) else "Unknown"
        color = colors["adult"] if cls == 0 else colors["calf"]
        
        draw_rectangle(ax, box, cls, label, color, gt_style)

    # clean up plot and save
    ax.axis('off')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def filter_overlapping_boxes(results, iou_threshold=0.8):
    filtered_results = []
    count_removed  = 0
    for i, (box1, score1, cls1, x_offset1, y_offset1) in enumerate(results):
        keep = True
        for j, (box2, score2, cls2, x_offset2, y_offset2) in enumerate(filtered_results):
            # Only compare boxes of the same class
            if cls1 == cls2:
                iou = calculate_iou(
                    [box1[0] + x_offset1, box1[1] + y_offset1, box1[2] + x_offset1, box1[3] + y_offset1],
                    [box2[0] + x_offset2, box2[1] + y_offset2, box2[2] + x_offset2, box2[3] + y_offset2]
                )
                if iou > iou_threshold:
                    count_removed += 1
                    keep = False
                    break
        if keep:
            filtered_results.append((box1, score1, cls1, x_offset1, y_offset1))
    return filtered_results

def predict(image, predictor, tile_size, overlap, use_filter=False):
    tiles, _, _ = tile_image(image, tile_size, overlap)
    results = []
    for tile, x, y in tiles:
        outputs = predictor(tile)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        for box, score, cls in zip(boxes, scores, classes):
            if score >= predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                results.append((box, score, cls, x, y))

    if use_filter:
        results = filter_overlapping_boxes(results, 0.6)
    return results


def load_predictor(config_file, model_weights, num_classes=2, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Set threshold for the model
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Update to match the number of classes
    predictor = DefaultPredictor(cfg)
    return predictor

def save_predictions_to_json(results, output_json_path):
    detections = []
    for box, score, cls, x_offset, y_offset in results:
        x1, y1, x2, y2 = box
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
                "category_id": int(cls),
            }
        )
    with open(output_json_path, "w") as f:
        json.dump(detections, f, indent=4)


def process_folder(cfg):
    # load the configs
    config_file = cfg["CONFIG_FILE"]
    model_weights = cfg["MODEL_WEIGHTS"]
    num_classes = cfg.get("NUM_CLASSES", 2)  # Default to 2 if not provided
    image_folder = cfg["TILE_TEST_FOLDER_PATH"]
    tile_size = cfg["TILE_SIZE"]
    overlap = cfg["OVERLAP"]
    use_filter = cfg["USE_FILTER"]
    plot_prediction = cfg["PLOT_PREDICTION"]
    output_folder = cfg["OUTPUT_FOLDER"]
    annotation_file = cfg["TILE_TEST_ANNOTATION_PATH"]

    # load the predictor
    predictor = load_predictor(config_file, model_weights, num_classes)

    # load annotations if provided
    with open(annotation_file) as f:
        annotations = json.load(f)

    # create output directories
    pred_output_folder = os.path.join(output_folder, "predictions")
    os.makedirs(os.path.join(pred_output_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(pred_output_folder, "json"), exist_ok=True)

    # get the image filenames from the folder
    image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for image_filename in tqdm(image_filenames, desc="Processing Images"):
        image_path = os.path.join(image_folder, image_filename)
        output_image_path = os.path.join(pred_output_folder, "image", f"{os.path.splitext(image_filename)[0]}_pred.png")
        output_json_path = os.path.join(pred_output_folder, "json", f"{os.path.splitext(image_filename)[0]}_pred.json")

        image = cv2.imread(image_path)

        # perform predictions on the image
        preds = predict(image, predictor, tile_size, overlap, use_filter)
        
        annotation = []
        for ann in annotations['annotations']:
            if ann["image_id"] == image_filename:
                # convert ground truth bbox from [x, y, width, height] to [x1, y1, x2, y2]
                gt_box = [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3]
                ]
                annotation.append([gt_box, ann["category_id"]])
                
                # iterate over each predicted bounding box in the preds
                for box2, _, _, x_offset, y_offset in preds:
                    pred_box = [
                        box2[0] + x_offset,
                        box2[1] + y_offset,
                        box2[2] + x_offset,
                        box2[3] + y_offset
                    ]

        # visualize the predictions if required
        if plot_prediction:
            if len(annotation) > 0:
                visualize_preds_with_gt(image, preds, annotation, output_image_path)

                
        # save the preds as a JSON file
        save_predictions_to_json(preds, output_json_path)
    
    print("All images processed")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, './../config.yaml')
    
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    process_folder(cfg)
