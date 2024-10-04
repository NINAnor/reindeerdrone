import gradio as gr
import torch
import os
import yaml
import cv2

from yaml import FullLoader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from dataset_utils import get_reindeer_dicts

# function to load the model
def load_model():
    dataset_name = "reindeer_test"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, './../config.yaml')
    
    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)
    
    img_dir = cfgP["TILE_TEST_FOLDER_PATH"]
    annotations_file = cfgP["TILE_TEST_ANNOTATION_PATH"]
    model_weights = cfgP["MODEL_WEIGHTS"]

    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, lambda: get_reindeer_dicts(img_dir, annotations_file))
        MetadataCatalog.get(dataset_name).set(thing_classes=["Adult", "Calf"])

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
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = predictor(image_bgr)

    # extract the bounding boxes and predictions
    v = Visualizer(image_bgr[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # get the output image with drawn bounding boxes
    output_image = v.get_image()[:, :, ::-1] 

    return output_image

# gradio interface
def gradio_interface(image):
    predictor, cfg = load_model()
    result_image = predict_and_visualize(image, predictor, cfg)
    return result_image

example_images_dir = os.path.join(os.path.dirname(__file__), "../assets/gradio_example_images")
example_images = [os.path.join(example_images_dir, img) for img in os.listdir(example_images_dir) if img.endswith(('png', 'jpg', 'jpeg'))]

# create Gradio interface
gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="numpy", label="Upload or Select an Example Image"),
    outputs="image",
    title="Reindeer Detection Model",
    description="Upload a photo of reindeer or select an example image to get predicted bounding boxes.",
    examples=example_images 
)

# launch the interface
if __name__ == "__main__":
    gr_interface.launch()