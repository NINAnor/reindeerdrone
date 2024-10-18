#!/usr/bin/env python3

import gradio as gr
import torch
import cv2
from pathlib import Path
import yaml

from yaml import FullLoader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from dataset_utils import get_reindeer_dicts
from functools import partial

def load_model():
    """ Function to load the model and configuration settings

    Returns:
        predictor (DefaultPredictor): The predictor object
        cfg (CfgNode): The configuration object
    """
    dataset_name = "reindeer_test"
    
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config.yaml"

    
    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)
    
    img_dir = cfgP["TILE_TEST_FOLDER_PATH"]
    annotations_file = cfgP["TILE_TEST_ANNOTATION_PATH"]
    model_weights = cfgP["MODEL_WEIGHTS"]

    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, partial(get_reindeer_dicts, img_dir=img_dir, annotations_file=annotations_file))
        MetadataCatalog.get(dataset_name).set(thing_classes=["Adult", "Calf"])

    # setting up the model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DATASETS.TEST = (dataset_name,)

    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def predict_and_visualize(image, predictor, cfg):
    """Predict and visualize bounding boxes on the input image using the specified model.

    Args:
        image (numpy.ndarray): The input image in RGB format as a NumPy array.
        predictor (DefaultPredictor): The Detectron2 predictor object used to make predictions.
        cfg (CfgNode): The Detectron2 configuration object containing model and dataset configurations.

    Returns:
        numpy.ndarray: The output image in RGB format with bounding boxes and instance predictions drawn.
    """
    outputs = predictor(image)

    # opencv uses BGR by default, but the visualizer expects RGB, so we reverse the channels
    v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_image = v.get_image()

    return output_image

def gradio_interface(image, predictor, cfg):
    """Gradio interface function to predict and visualize bounding boxes.

    This function is used as the callback function for the Gradio interface. 
    It takes an input image, passes it to the prediction model, and returns 
    the image with bounding boxes drawn around the detected reindeer.

    Args:
        image (numpy.ndarray): The input image in RGB format as a NumPy array.
        predictor (DefaultPredictor): The Detectron2 predictor object used to make predictions.
        cfg (CfgNode): The Detectron2 configuration object containing model and dataset configurations.

    Returns:
        numpy.ndarray: The output image in RGB format with bounding boxes and predictions drawn.
    """
    result_image = predict_and_visualize(image, predictor, cfg)
    return result_image

def main():
    predictor, cfg = load_model()
    interface_fn = partial(gradio_interface, predictor=predictor, cfg=cfg)

    # define example images
    example_images_drone_dir = Path(__file__).resolve().parent.parent / "assets" / "gradio_example_images" / "drone_images"
    example_images_heli_dir = Path(__file__).resolve().parent.parent / "assets" / "gradio_example_images" / "helicopter_images"
    example_images = list(example_images_drone_dir.glob("*.png")) + list(example_images_heli_dir.glob("*.png")) 
    
    # create gradio interface
    gr_interface = gr.Interface(
        fn=interface_fn,
        inputs=gr.Image(type="numpy", label="Upload or select an example image"),
        outputs="image",
        title="Reindeer detection model",
        description="Upload a photo of reindeer or select an example image to get predicted bounding boxes.",
        examples=[str(img) for img in example_images]
    )

    gr_interface.launch()

if __name__ == "__main__":
    main()
