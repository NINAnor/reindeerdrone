import marimo

__generated_with = "0.9.4"
app = marimo.App(
    width="medium",
    app_title="Model evaluation",
    layout_file="layouts/recall-precision.slides.json",
)


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
    model_weights = "/home/taheera.ahmed/code/reindeerdrone/output/00_default_augs/model_final.pth"
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
    iou_threshold = 0.2  # Threshold for a valid detection
    false_positive_images = {}  # To store false positives per image
    false_negative_images = {}

    # images which are outliers (produces)
    bad_images = (
        "DSC09929_tile44.png",
        "DSC01033_tile33.png",
        "DSC01033_tile6.png",
        "DSC01033_tile55.png",
        "DSC09874_tile42.png",
        "DSC09929_tile26.png",
        "DSC01033_tile67.png",
        "DSC01033_tile75.png",
        #"DSC01026_tile35.png",
        #"DSC09929_tile39.png",
        #"DSC09874_tile35.png",
        # false negatives
        "DSC09929_tile32.png",
        "DSC00949_tile15.png",
        "DSC09874_tile2.png",
        "DSC09929_tile45.png",
        "DSC09929_tile25.png",
        "DSC09874_tile1.png",
        #"DSC09874_tile28.png",
        #"DSC00949_tile8.png",
        #"DSC00949_tile9.png",
        #"DSC09929_tile38.png",
        #"DSC01026_tile21.png",
        #"DSC09874_tile9.png"
        )


    for img_filename in tqdm(image_filenames, desc="Processing Images"):
        if img_filename in bad_images:
            print(f"skipping {img_filename}")
            continue
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
        if img_filename not in false_negative_images:
            false_negative_images[img_filename] = 0  # Initialize to 0 if not present

        # track matched ground truths to detect false negatives
        matched_gt_ids = set()

        # IOU matching and classification evaluation
        for pred in preds:
            pred_box = pred[0]  # Predicted bounding box
            pred_class = pred[2]  # Predicted class (calf or adult)
            # find the best matching annotation (ground truth) by IoU
            best_iou = 0
            best_gt = None
            for gt in img_annotations:
                gt_box = gt_to_pred_format(gt['bbox'])  # Convert ground truth bbox to same format as pred bbox
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            # if the best IoU is above the threshold, consider it a valid match
            if best_iou >= iou_threshold and best_gt is not None:
                y_true.append(best_gt['category_id'])
                y_pred.append(pred_class)
                matched_gt_ids.add(best_gt['id'])  # Mark this ground truth as matched
            else:
                # no matching ground truth, count this as a false positive
                y_true.append(None)  # no matching ground truth
                y_pred.append(pred_class)  # false positive prediction
                false_positive_count += 1
                false_positive_images[img_filename] += 1  # Increment false positives for this image
        # check for false negatives (missed ground truths)
        for gt in img_annotations:
            if gt['id'] not in matched_gt_ids:
                y_true.append(gt['category_id'])  # This ground truth was missed
                y_pred.append(None)  # No prediction for this ground truth
                false_negative_images[img_filename] += 1
    return (
        ann_adult_count,
        ann_calf_count,
        bad_images,
        best_gt,
        best_iou,
        compare_dict,
        false_negative_images,
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
        matched_gt_ids,
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
    mo.md(
        r"""
        # Introduction

        Talk about model performance with respect to the questions you've had :D
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Just a quick overview

        There are many images where the **count** of annotated vs. predicted images are correct. Being to the left means there are more predicted bounding boxes and being to the right means there are more annotated.
        """
    )
    return


@app.cell(hide_code=True)
def __(alt, compare_dict, pd):
    _data = []
    for _img, _counts in compare_dict.items():
        _data.append({
            'Image': _img,
            'Difference Adult': _counts['ann_adult_count'] - _counts['pred_adult_count'],
            'Difference Calf': _counts['ann_calf_count'] - _counts['pred_calf_count'], 
        })

    _df = pd.DataFrame(_data)

    # Reshape the data for Altair plotting
    _df_melted = _df.melt(id_vars=['Image'], var_name='Class', value_name='Difference')

    # Create histogram with bins
    _histogram = alt.Chart(_df_melted).mark_bar().encode(
        x=alt.X('Difference:Q', bin=alt.Bin(maxbins=15), title="Difference"),
        y='count()',
        color='Class:N',
        tooltip=['Class', 'count()', 'Difference:Q']
    ).properties(
        width=800,
        height=400,
        title='Histogram of Differences Between Annotated - Predicted Counts'
    )

    _histogram.display()
    return


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

        The measure of agreement between the model and observer detection can be expressed as the **recall**, which represents the probability that the model will detect a reindeer given that the observer has detected it. Recall answers the question: "Out of all the actual positive instances, how many did the model correctly identify?"

        We will also loook into **precision** which is a metric which answers the question: "Out of all the instances the model predicted as positive, how many were actually positive?"
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q2

        _What is the probability that the model will categorize an identified reindeer correctly (calf and adult)?_

        The metrics which have been used to answer this are the precision and recall.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Confusion matrix

        First let me introduce the confusion matrix. This tells the whole story about recall and precision. Shows insight into the classification done by the model. It looks at each bounding box and its class and compares it to the ground truth.

        _Hehe hover over and explain the numbers_
        """
    )
    return


@app.cell(hide_code=True)
def __(alt, confusion_matrix, pd, y_pred, y_true):
    def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
        # Replace None values in y_true and y_pred with 'None' for better compatibility with Altair
        y_true = ['None' if x is None else x for x in y_true]
        y_pred = ['None' if x is None else x for x in y_pred]

        # Generate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels + ['None'])
        # Create a DataFrame from the confusion matrix
        cm_df = pd.DataFrame(cm, index=["Adult", "Calf", "None"], columns=["Adult", "Calf", "None"])

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
        return cm

    cm = plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title="Reindeer Detection Confusion Matrix")
    return cm, plot_confusion_matrix


@app.cell(hide_code=True)
def __(classification_report, mo, y_pred, y_true):
    y_true_with_none = [2 if yt is None else yt for yt in y_true]
    y_pred_with_none = [2 if yp is None else yp for yp in y_pred]

    _report = classification_report(
        y_true_with_none, y_pred_with_none, 
        labels=[0, 1, 2],  # Include the label for 'None' (2)
        target_names=["Adult", "Calf", "None"]
    )
    print(f"{_report}")
    mo.md("""
    ### Precision and recall

    The model performs best on the "Adult" class, with a recall of 83%, meaning it correctly identifies 83% of actual adult reindeer. However, the precision is relatively low at 63%, indicating many of the predicted "Adult" reindeers are misclassifications.

    For the "Calf" class, the model has a precision of 66% and a recall of 47%. This indicates that when the model predicts a reindeer as a "Calf," it is more often correct, but it struggles to capture all calf instances, detecting only 47% of them. 

    This might be due to the fact that there are fewer calves and they vary more in colors, therefore it makes sense that the metrics are much lower for "Calf".
    """)
    return y_pred_with_none, y_true_with_none


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q3
        _Overall, what is the most parsimonious identification model. The one with reindeer only, or the one distinguishing reindeer into two categories?_

        We haven't trained a model for only reindeer detection, but we don't believe this would make a huge difference. This is because we will then introduce more variation to the general reindeer class.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q4
        _Can anything meaningful be said about the models’ probability to erroneously identify a reindeer? I.e., a “model reindeer” that an observer would not identify as a reindeer (rock, tree trunk, etc.)._

        In order to investigate this, we have looked into the false negatives and false positives, this means where the model has predicted wrong with respect to the annotations.

        * False positive per image: Imaginary reindeer (the model sees reindeer which are not there)
        * False negative per image: Missing reindeer (the model doesn't see some reindeer which are actually there)
        """
    )
    return


@app.cell
def __(false_negative_images, false_positive_images, pd):
    false_positive_df = pd.DataFrame(list(false_positive_images.items()), columns=['Image', 'False Positives'])
    false_negative_df = pd.DataFrame(list(false_negative_images.items()), columns=['Image', 'False Negatives'])
    return false_negative_df, false_positive_df


@app.cell(hide_code=True)
def __(alt, false_negative_df, false_positive_df, pd):

    # Merge the two dataframes on 'Image'
    _merged_df = pd.merge(false_positive_df, false_negative_df, on='Image')

    # Create a bar chart with both False Positives and False Negatives
    false_positive_hist = alt.Chart(_merged_df).mark_bar().encode(
        alt.X('False Positives:Q', bin=alt.Bin(maxbins=10), title='Number of False Positives'),
        alt.Y('count()', title='Frequency'),
        color=alt.value('purple'),
        tooltip=['False Positives', 'count()']
    ).properties(
        title='Distribution of False Positives Across Images',
        width=450,
        height=400
    )

    false_negative_hist = alt.Chart(_merged_df).mark_bar().encode(
        alt.X('False Negatives:Q', bin=alt.Bin(maxbins=10), title='Number of False Negatives'),
        alt.Y('count()', title='Frequency'),
        color=alt.value('blue'),
        tooltip=['False Negatives', 'count()']
    ).properties(
        title='Distribution of False Negatives Across Images',
        width=450,
        height=400
    )

    # Concatenate the two histograms side by side
    combined_chart = alt.hconcat(false_positive_hist, false_negative_hist)

    combined_chart
    return combined_chart, false_negative_hist, false_positive_hist


@app.cell(hide_code=True)
def __(Image, false_positive_df, math, mo, patches, plt, predictions):
    top_false_positives = false_positive_df.sort_values(by='False Positives', ascending=False).head(4)

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
        plt.savefig("./")
        plt.show()

    # Call the function to plot all images in a grid with 2 columns
    plot_images_in_grid(plot_preds)
    mo.md("""
    ### Looking at the images with imaginary reindeers (false positive)

    The worst examples with many false positives. As you can see the reindeers here varies in size due to the difference in elevation from the ground vs. the drone itself. If there are objects with the same size, shape and colors (such as tree trunks and rocks, the model has difficulties with extinguishing these. However we are using a special type of model which should take varying sizes of the reindeers into consideration. 

    This is most likely due to the lack of training data.

    """)
    return (
        convert_to_coco_format,
        data_path,
        plot_images_in_grid,
        plot_preds,
        top_false_positives,
        top_false_positives_list,
    )


@app.cell(hide_code=True)
def __(data_path, false_negative_df, mo, plot_images_in_grid, predictions):
    top_false_negatives = false_negative_df.sort_values(by='False Negatives', ascending=False).head(4)

    top_false_negatives_list = top_false_negatives['Image'].values.tolist()

    _plot_preds = []
    for _img_id in top_false_negatives_list:
        for _pred in predictions:
            if (_img_id == _pred["image_id"]):
                _new_pred = _pred
                _new_pred['data_path'] = data_path + _img_id
                _plot_preds.append(_new_pred)

    # Call the function to plot all images in a grid with 2 columns
    plot_images_in_grid(_plot_preds)
    mo.md("""
    ### Looking at the images with missing prediction (false negative)

    Here we look at which annotations the model has missed, and has not been predicted. _Talk about each image_. In general one can see that the reindeers which are very small has been missed. This again might be due to the variation in height.
    """)
    return top_false_negatives, top_false_negatives_list


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### In general the model seems to be struggling with the variation in sizes of the reindeers, which is caused by the varying elevation of the drone.""")
    return


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
def __(convert_to_coco_format):
    def fix_preds(predictions):
        count_id = 0
        for _preds in predictions:
            _img_id = _preds['image_id']
            temp_pred_bbox = {}
            temp_pred_bboxs = []
            for _bbox in _preds['preds']:
                temp_pred_bbox['id'] = count_id
                _x, _y, _width, _height = convert_to_coco_format(_bbox[0])
                temp_pred_bbox['bbox'] = [_x, _y, _width, _height]
                temp_pred_bbox['area'] = _width * _height
                temp_pred_bbox['pred_category_id'] = _bbox[2]
                temp_pred_bbox['pred_prob'] = _bbox[1]
                temp_pred_bbox['image_id'] = _img_id
                temp_pred_bboxs.append(temp_pred_bbox)
                count_id +=1
            _preds['preds'] = temp_pred_bboxs


    return (fix_preds,)


@app.cell(hide_code=True)
def __(pd, predictions):
    _bbox_areas = []

    category_mapping = {
        0: "adult",
        1: "calf"
    }

    for _item in predictions:
        for _pred in _item.get("preds", []):
            _bbox = _pred.get("bbox")
            if _bbox:
                _area = _bbox[2] * _bbox[3]  # Width * Height
                _category = category_mapping.get(_pred.get("pred_category_id", -1), "unknown")
                _bbox_areas.append({"type": "prediction", "class": _category, "area": _area})
        
        for _annotation in _item.get("annotations", []):
            _bbox = _annotation.get("bbox")
            if _bbox:
                _area = _bbox[2] * _bbox[3]  # Width * Height
                _category = category_mapping.get(_annotation.get("category_id", -1), "unknown")
                _bbox_areas.append({"type": "annotation", "class": _category, "area": _area})

    # Convert the extracted data into a DataFrame
    bbox_areas_df = pd.DataFrame(_bbox_areas)
    bbox_areas_df.groupby(['type', 'class'])['area'].describe()
    return bbox_areas_df, category_mapping


@app.cell(hide_code=True)
def __(alt, bbox_areas_df):
    _kde_chart = alt.Chart(bbox_areas_df).transform_density(
        'area',
        groupby=['type'],
        as_=['area', 'density']
    ).mark_area(opacity=0.5).encode(
        x='area:Q',
        y='density:Q',
        color='type:N'
    ).properties(
        title="KDE of Bounding Box Sizes for Predictions and Annotations",
        width=800
    )

    # Create histogram plot
    _histogram_chart = alt.Chart(bbox_areas_df).mark_bar(opacity=0.5).encode(
        alt.X('area:Q', bin=alt.Bin(maxbins=40), title='Bounding Box Area'),
        alt.Y('count()', title='Count'),
        alt.Color('type:N', title='Type')
    ).properties(
        title="Histogram of Bounding Box Sizes for Predictions and Annotations",
        width=800
    )

    # Show both charts
    _kde_chart & _histogram_chart
    return


@app.cell(hide_code=True)
def __(alt, bbox_areas_df):

    _kde_chart = alt.Chart(bbox_areas_df).transform_density(
        'area',
        groupby=['type', 'class'],  # Group by both type and class (calf/adult)
        as_=['area', 'density']
    ).mark_area(opacity=0.5).encode(
        x='area:Q',
        y='density:Q',
        color=alt.Color('class:N', title='Class (Calf/Adult)'),  # Color by class
        row='type:N'  # Separate rows for predictions and annotations
    ).properties(
        title="KDE of Bounding Box Sizes by Class (Calf/Adult) for Predictions and Annotations",
        width=800,
    )

    # Create histogram plot with legend for each class and type (annotations vs. predictions)
    _histogram_chart = alt.Chart(bbox_areas_df).mark_bar(opacity=0.5).encode(
        alt.X('area:Q', bin=alt.Bin(maxbins=30), title='Bounding Box Area'),
        alt.Y('count()', title='Count'),
        alt.Color('class:N', title='Class (Calf/Adult)'),  # Color by class
        row='type:N'  # Separate rows for predictions and annotations
    ).properties(
        title="Histogram of Bounding Box Sizes by Class (Calf/Adult) for Predictions and Annotations",
        width=800,
    )

    # Show both charts with the legend
    _kde_chart & _histogram_chart
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Q6

        _I have asked for some pictures taken from helicopters during the regular calf surveys. It would be interesting to run these pictures through the mature model to see how well it manages to detect individuals and distinguish calves from other individuals._

        This did not do well, sadly. Referring to the application. This is because the different perspective, and the reindeer having different shapes than what it has been trained on.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Final words

        ## Moving forward

        * IR drone photos will make this task a lot easier -- the color variation of each calf has made it difficult, and also the varying size of all the reindeers caused by the varying elevetion.
        * Do we want more seasons?
        * Do we want more training data?


        ## Bottleneck

        * Limited budget and couldn't afford more data
        """
    )
    return


@app.cell(hide_code=True)
def __(cm, mo, np, plt):
    # Calculate recall for each class
    recall_per_class = np.diag(cm) / np.sum(cm, axis=1)

    # Total instances for each class (TP + FN)
    total_per_class = np.sum(cm, axis=1)

    # Function to calculate SE and CI
    def calculate_se_and_ci(recall, total, confidence_level=0.95):
        # Calculate Standard Error (SE)
        se = np.sqrt((recall * (1 - recall)) / total)

        # Calculate the z-score for the confidence level (e.g., 1.96 for 95% CI)
        z = 1.96

        # Calculate Confidence Interval (CI)
        lower_bound = recall - z * se
        upper_bound = recall + z * se

        return se, (max(0, lower_bound), min(1, upper_bound))  # CI should be between 0 and 1

    # Calculate SE and CI for each class
    classes = ['Adult', 'Calf', 'None']
    recall_values = []
    se_values = []
    ci_values = []

    for i, recall in enumerate(recall_per_class):
        se, ci = calculate_se_and_ci(recall, total_per_class[i])
        recall_values.append(recall)
        se_values.append(se)
        ci_values.append(ci)

    fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))  # 1 row, 2 columns, wider figure

    # Plot Recall with CI
    recall_means = np.array(recall_values)
    lower_bounds = np.array([ci[0] for ci in ci_values])
    upper_bounds = np.array([ci[1] for ci in ci_values])
    yerr = [recall_means - lower_bounds, upper_bounds - recall_means]

    _ax1.errorbar(classes, recall_means, yerr=yerr, fmt='o', capsize=5, label="Recall with 95% CI")
    _ax1.set_xlabel("Classes")
    _ax1.set_ylabel("Recall")
    _ax1.set_title("Recall and Confidence Intervals for Each Class")
    _ax1.legend()

    # Plot Standard Error
    _ax2.bar(classes, se_values, color='skyblue')
    _ax2.set_xlabel("Classes")
    _ax2.set_ylabel("Standard Error")
    _ax2.set_title("Standard Error for Each Class")

    plt.tight_layout()
    plt.show()

    mo.md("""
    The tight CI and low SE for the "Adult" class suggest the model performs reliably for adults, though the precision might still need improvement.

    The wider CI and higher SE for the "Calf" class indicate the model struggles with identifying calves, and there is greater uncertainty around its performance.

    """)
    return (
        calculate_se_and_ci,
        ci,
        ci_values,
        classes,
        fig,
        i,
        lower_bounds,
        recall,
        recall_means,
        recall_per_class,
        recall_values,
        se,
        se_values,
        total_per_class,
        upper_bounds,
        yerr,
    )


if __name__ == "__main__":
    app.run()
