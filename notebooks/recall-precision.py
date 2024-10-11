import marimo

__generated_with = "0.9.4"
app = marimo.App(width="medium", app_title="Model evaluation")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Answer questions from Torkild and Vebjørn 🕺🕺""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Importing libraries""")
    return


@app.cell
def __():
    import os
    import cv2
    import yaml
    import json
    import math

    import marimo as mo
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import altair as alt
    import pandas as pd
    import matplotlib.patches as patches

    from PIL import Image
    from sklearn.metrics import confusion_matrix
    from tqdm import tqdm
    from yaml import FullLoader
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    return (
        DefaultPredictor,
        FullLoader,
        Image,
        alt,
        classification_report,
        confusion_matrix,
        cv2,
        get_cfg,
        json,
        math,
        mo,
        model_zoo,
        np,
        os,
        patches,
        pd,
        plt,
        sns,
        tqdm,
        yaml,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Setting up configurations""")
    return


@app.cell
def __(FullLoader, __file__, os, yaml):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, './../config.yaml')

    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)
    return cfgP, config_path, current_dir, f


@app.cell
def __(cfgP):
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    model_weights = "/home/taheera.ahmed/code/reindeerdrone/output/00_default_augs/best_val_loss_model.pth"
    num_classes = cfgP.get("NUM_CLASSES", 2)  # Default to 2 if not provided
    image_folder = cfgP["TILE_TEST_FOLDER_PATH"]
    tile_size = cfgP["TILE_SIZE"]
    overlap = cfgP["OVERLAP"]
    use_filter = cfgP["USE_FILTER"]
    plot_prediction = cfgP["PLOT_PREDICTION"]
    output_folder = cfgP["OUTPUT_FOLDER"]
    annotation_file = cfgP["TILE_TEST_ANNOTATION_PATH"]
    return (
        annotation_file,
        config_file,
        image_folder,
        model_weights,
        num_classes,
        output_folder,
        overlap,
        plot_prediction,
        tile_size,
        use_filter,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Loading model""")
    return


@app.cell
def __(
    DefaultPredictor,
    config_file,
    get_cfg,
    model_weights,
    model_zoo,
    num_classes,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for the model
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Update to match the number of classes
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Load annotations and find test image filenames""")
    return


@app.cell(hide_code=True)
def __(annotation_file, image_folder, json, os):
    with open(annotation_file) as anno_file:
        annotations = json.load(anno_file)

    image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return anno_file, annotations, image_filenames


@app.cell(hide_code=True)
def __(cv2, image_filenames, image_folder, os, plt):
    _image_filename = image_filenames[392]

    _image_path = os.path.join(image_folder, _image_filename)
    _image = cv2.imread(_image_path)
    height, width, _ = _image.shape
    if len(_image.shape) == 3 and _image.shape[2] == 3:
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    _fig, _ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    _ax.axis('off')
    _ax.set_title(_image_filename)
    _ax.imshow(_image)
    return height, width


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Helper functions

        Just a dump of various functions used throughout this notebook 💩
        """
    )
    return


@app.cell(hide_code=True)
def __(cv2):
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

    def gt_to_pred_format(gt_bbox):
        x, y, width, height = gt_bbox
        x2 = x + width
        y2 = y + height
        return [x, y, x2, y2]

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
    return (
        calculate_iou,
        filter_overlapping_boxes,
        gt_to_pred_format,
        predict,
        tile_image,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Predicting 🤖""")
    return


@app.cell(hide_code=True)
def __(
    annotations,
    calculate_iou,
    cv2,
    gt_to_pred_format,
    image_filenames,
    image_folder,
    os,
    overlap,
    predict,
    predictor,
    tile_size,
    tqdm,
    use_filter,
):
    compare_dict = {}
    predictions = []
    y_true = []  # Ground truth class (calf or adult)
    y_pred = []  # Predicted class (calf or adult)
    false_positive_count = 0  # To count false positives
    iou_threshold = 0.5  # Threshold for a valid detection
    false_positive_images = {}  # To store false positives per image

    for img_filename in tqdm(image_filenames, desc="Processing Images"):
        img_path = os.path.join(image_folder, img_filename)
        img = cv2.imread(img_path)

        # Get predictions using the predict function
        preds = predict(img, predictor, tile_size, overlap, use_filter)

        # Find annotations for the current image
        img_annotations = [ann for ann in annotations['annotations'] if ann["image_id"] == img_filename]

        predictions.append(
            {
                "image_id": img_filename,
                "preds": preds,
                "annotations": img_annotations  
            }
        )

        ann_adult_count = sum(1 for ann in img_annotations if ann["category_id"] == 0)
        ann_calf_count = sum(1 for ann in img_annotations if ann["category_id"] != 0)

        pred_adult_count = sum(1 for pred in preds if pred[2] == 0)
        pred_calf_count = sum(1 for pred in preds if pred[2] != 0)

        # Store counts in compare_dict
        compare_dict[img_filename] = {
            'ann_adult_count': ann_adult_count,
            'ann_calf_count': ann_calf_count,
            'pred_adult_count': pred_adult_count,
            'pred_calf_count': pred_calf_count
        }

        if img_filename not in false_positive_images:
            false_positive_images[img_filename] = 0  # Initialize to 0 if not present

        # IoU-based matching and classification evaluation
        for pred in preds:
            pred_box = pred[0]  # Predicted bounding box
            pred_class = pred[2]  # Predicted class (calf or adult)

            # Find the best matching annotation (ground truth) by IoU
            best_iou = 0
            best_gt = None
            for gt in img_annotations:
                gt_box = gt_to_pred_format(gt['bbox'])  # Convert ground truth bbox to same format as pred bbox
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            # If the best IoU is above the threshold, consider it a valid match
            if best_iou >= iou_threshold and best_gt is not None:
                y_true.append(best_gt['category_id'])
                y_pred.append(pred_class)
            else:
                # No matching ground truth, so count this as a false positive
                false_positive_count += 1
                false_positive_images[img_filename] += 1  # Increment false positives for this image
    return (
        ann_adult_count,
        ann_calf_count,
        best_gt,
        best_iou,
        compare_dict,
        false_positive_count,
        false_positive_images,
        gt,
        gt_box,
        img,
        img_annotations,
        img_filename,
        img_path,
        iou,
        iou_threshold,
        pred,
        pred_adult_count,
        pred_box,
        pred_calf_count,
        pred_class,
        predictions,
        preds,
        y_pred,
        y_true,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Answer questions ⭐""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q1
        _A measure of agreement between model and observer detection and classification. In other words, what is the average probability (including CI, SE or similar) that the model will detect a reindeer (of any category) given that an observer has detected the reindeer?_
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q2

        _What is the probability that the model will categorize an identified reindeer correctly (calf and adult)?_
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""

        ### Confusion matrix 

        #### Upper row
        By starting in the upper row one can see that 316 reindeer has been predicted as "Adult", and is annotated as an   "Adult" as well. This is known as "True Positives". They are correctly classified by the model.

        However there are 13 reindeer which has been misclassified as a "Calf", when they are really "Adults", these are known as "False Positives". 

        #### Lower row
        There are 57 predictions which have been incorrectly predicted as "Adult" when they were annotated as "Calf". Lastly, there are 84 reindeers which have been correctly classified as "Calf" when they have been annotated as "Calf".

        """
    )
    return


@app.cell(hide_code=True)
def __(alt, confusion_matrix, pd, y_pred, y_true):
    def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create a DataFrame from the confusion matrix
        cm_df = pd.DataFrame(cm, index=["Adult", "Calf"], columns=["Adult", "Calf"])
        
        # Convert the DataFrame to long format for Altair
        cm_long = cm_df.reset_index().melt(id_vars='index')
        cm_long.columns = ['True Label', 'Predicted Label', 'Count']
        
        # Create the heatmap using Altair
        heatmap = alt.Chart(cm_long).mark_rect().encode(
            alt.X('Predicted Label:N', title='Predicted Label'),
            alt.Y('True Label:N', title='True Label'),
            alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
            tooltip=['True Label:N', 'Predicted Label:N', 'Count:Q']
        ).properties(
            title=title,
            width=800,
            height=500
        )
        
        # Overlay text to show the counts in each square
        text = heatmap.mark_text(baseline='middle', size=20).encode(
            text='Count:Q',
            color=alt.condition(
                alt.datum.Count > 0,  # If the count is greater than 0, display in black, otherwise white
                alt.value('black'),
                alt.value('white')
            )
        )
        
        # Combine heatmap and text
        heatmap_with_text = heatmap + text
        
        # Display the heatmap with numbers
        heatmap_with_text.show()

    # Example usage after getting y_true and y_pred from the previous code:


    plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title="Reindeer Detection Confusion Matrix")

    return (plot_confusion_matrix,)


@app.cell(hide_code=True)
def __(classification_report, mo, y_pred, y_true):
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Adult", "Calf"])
    print(f"{report}")
    mo.md("""
    ### Precision and recall

    The model performs very well in identifying Adult reindeers, as indicated by the high precision (84.7%) and recall (96.0%) for the "Adult" class.

    The model struggles a bit more with identifying Calf reindeers, with a lower recall (59.6%). This suggests that many calves are being incorrectly classified as adults, as evidenced by the relatively high number of false positives (57).

    Fewer calves and they vary more in colors, therefore it makes sense that the metrics are much lower for "Calf".
    """)
    return (report,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q3
        _Overall, what is the most parsimonious identification model. The one with reindeer only, or the one distinguishing reindeer into two categories?_

        We haven't trained a model for only reindeer detection, but we don't believe this would make a huge difference. This is because the model mainly fails on classification, and not on actually detecting the reindeer.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q4
        _Can anything meaningful be said about the models’ probability to erroneously identify a reindeer? I.e., a “model reindeer” that an observer would not identify as a reindeer (rock, tree trunk, etc.)._
        """
    )
    return


@app.cell
def __(alt, false_positive_images, pd):
    false_positive_df = pd.DataFrame(list(false_positive_images.items()), columns=['Image', 'False Positives'])

    _chart = alt.Chart(false_positive_df).mark_bar().encode(
        alt.X('False Positives:Q', bin=alt.Bin(maxbins=10), title='Number of False Positives'),  # Binned into 10 buckets
        y=alt.Y('count()', title='Number of Images'),
        tooltip=['False Positives', 'count()']
    ).properties(
        title='Distribution of False Positives Across Images',
        width=800,
        height=400
    )

    _chart.show()
    return (false_positive_df,)


@app.cell
def __(false_positive_df):
    false_positive_df
    return


@app.cell(hide_code=True)
def __(Image, false_positive_df, math, mo, patches, plt, predictions):
    top_false_positives = false_positive_df.sort_values(by='False Positives', ascending=False).head(6)

    top_false_positives_list = top_false_positives['Image'].values.tolist()
    set(top_false_positives_list)

    data_path = '/home/taheera.ahmed/data/reindeerdrone/tiles/test/'

    plot_preds = []
    for _img_id in top_false_positives_list:
        for _pred in predictions:
            if (_img_id == _pred["image_id"]):
                _new_pred = _pred
                _new_pred['data_path'] = data_path + _img_id
                plot_preds.append(_new_pred)

    def convert_to_coco_format(bbox):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return [x_min, y_min, width, height]

    def plot_images_in_grid(plot_preds):
        num_images = len(plot_preds)
        num_columns = 2
        num_rows = math.ceil(num_images / num_columns)

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 6))
        axes = axes.flatten()  # Flatten the axes array for easier iteration
        red_patch = patches.Patch(color='red', label='Annotation')
        blue_patch = patches.Patch(color='blue', label='Prediction')

        for i, _pred in enumerate(plot_preds):
            img = Image.open(_pred['data_path'])
            ax = axes[i]
            ax.set_title(_pred['image_id'])
            # Display the image
            ax.imshow(img)
            # Add bounding boxes for annotations
            for ann_bbox in _pred['annotations']:
                x, y, width, height = ann_bbox['bbox']
                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            # Add bounding boxes for predictions
            for pred_bbox in _pred['preds']:
                x, y, width, height = convert_to_coco_format(pred_bbox[0])
                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

            # Remove axes for a cleaner look
            ax.axis('off')

            if i == 0:
                ax.legend(handles=[red_patch, blue_patch], loc='upper right')

        # Remove any extra subplots that don't have images
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    # Call the function to plot all images in a grid with 2 columns
    plot_images_in_grid(plot_preds)
    mo.md("""
    The worst examples with many false positives""")
    return (
        convert_to_coco_format,
        data_path,
        plot_images_in_grid,
        plot_preds,
        top_false_positives,
        top_false_positives_list,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q5
        _Is there any information from the model that could be used to e.g., classify the various source pictures/tiles as “distant” or “close-ups”. For instance, average absolute size of the box placed around adult reindeer by the model or by the observer? This could then be used as a categorical index for flight height. This would be helpful if we should describe some of the current limitations._

        This would be a quite difficult task I think. We could look at the average bounding box size for each image, but I think the sizes of the bounding boxes varies too much per annotation per image.. 
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q6

        _I have asked for some pictures taken from helicopters during the regular calf surveys. It would be interesting to run these pictures through the mature model to see how well it manages to detect individuals and distinguish calves from other individuals._

        This did not do well, sadly. Referring to the application.
        """
    )
    return


if __name__ == "__main__":
    app.run()
