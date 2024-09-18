import os
import json
import yaml
import numpy as np
import cv2
import random
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase, launch, default_argument_parser
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

from yaml import FullLoader

def normalize_image_id(image_id):
    # Normalizes image_id by stripping any directory structure
    return os.path.splitext(os.path.basename(image_id))[0]

def split_dataset(annotations_file, split_ratio=0.8):
    # Load annotations
    with open(annotations_file) as f:
        coco = json.load(f)

    # Create a mapping of image file names (without extension) to their annotations
    image_annotations = {normalize_image_id(img['file_name']): img['id'] for img in coco['images']}
    annotations = {img_id: [] for img_id in image_annotations.values()}

    for anno in coco['annotations']:
        norm_image_id = normalize_image_id(anno['image_id'])
        if norm_image_id in annotations:
            annotations[norm_image_id].append(anno)

    # Split images into train and val sets
    image_files = list(image_annotations.keys())
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    train_annotations = [{'image_id': image_annotations[file], 'annotations': annotations[image_annotations[file]]} for file in train_files]
    val_annotations = [{'image_id': image_annotations[file], 'annotations': annotations[image_annotations[file]]} for file in val_files]

    return train_files, val_files, train_annotations, val_annotations

def get_reindeer_dicts(tile_dir, annotations):
    dataset_dicts = []
    for idx, anno in enumerate(annotations):
        record = {}
        filename = os.path.join(tile_dir, f"{anno['image_id']}.png")
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        objs = []
        for bbox in anno['annotations']:
            obj = {
                "bbox": bbox["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": bbox["category_id"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


class EarlyStoppingHook(HookBase):
    def __init__(self, patience=3, metric="bbox/AP"):
        self.patience = patience
        self.metric = metric
        self.best_metric = None
        self.num_bad_epochs = 0

    def after_step(self):
        if self.trainer.iter + 1 == self.trainer.max_iter:
            results = self.trainer.storage.latest()
            current_metric = results[self.metric]
            if self.best_metric is None or current_metric > self.best_metric:
                self.best_metric = current_metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                print(f"Early stopping triggered at iteration {self.trainer.iter + 1}")
                self.trainer.iter = self.trainer.max_iter  # Stop training


class ReindeerTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def __init__(self, cfg):
        super().__init__(cfg)
        # self.register_hooks([EarlyStoppingHook(patience=3, metric='bbox/AP')])


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x"
    )
    cfg.DATASETS.TRAIN = ("reindeer_train",)
    cfg.DATASETS.TEST = ("reindeer_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000

    # Assuming you have two classes: "Adult" and "Calf"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Set to 2 classes if you have both "Adult" and "Calf"
    cfg.OUTPUT_DIR = "./output"
    return cfg

def main(args):
    setup_logger()
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open("./config.yaml") as f:
        cfgP = yaml.load(f, Loader=FullLoader)

    # Split the dataset
    img_dir = cfgP["TILE_FOLDER_PATH"]
    annotations_file = cfgP["TILE_ANNOTATION_PATH"]
    train_files, val_files, train_annotations, val_annotations = split_dataset(
        annotations_file
    )

    # Register the dataset
    DatasetCatalog.register(
        "reindeer_train", lambda: get_reindeer_dicts(img_dir, train_annotations)
    )
    # Set thing_classes to the actual categories in your dataset
    MetadataCatalog.get("reindeer_train").set(thing_classes=["Adult", "Calf"])

    DatasetCatalog.register(
        "reindeer_val", lambda: get_reindeer_dicts(img_dir, val_annotations)
    )
    MetadataCatalog.get("reindeer_val").set(thing_classes=["Adult", "Calf"])

    # Initialize trainer
    trainer = ReindeerTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Start training
    trainer.train()

    # Evaluate the model
    evaluator = COCOEvaluator("reindeer_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "reindeer_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)



def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":

    invoke_main()
