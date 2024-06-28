import os
import json
import yaml
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase, launch, default_argument_parser
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

from dataset import get_reindeer_dicts, split_dataset

from yaml import FullLoader

class EarlyStoppingHook(HookBase):
    def __init__(self, patience=3, metric='bbox/AP'):
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
        #self.register_hooks([EarlyStoppingHook(patience=3, metric='bbox/AP')])

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("reindeer_train",)
    cfg.DATASETS.TEST = ("reindeer_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class (reindeer)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
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
    train_files, val_files, train_annotations, val_annotations = split_dataset(img_dir, annotations_file)

    # Register the dataset
    DatasetCatalog.register("reindeer_train", lambda: get_reindeer_dicts(img_dir, train_annotations))
    MetadataCatalog.get("reindeer_train").set(thing_classes=["reindeer"])

    DatasetCatalog.register("reindeer_val", lambda: get_reindeer_dicts(img_dir, val_annotations))
    MetadataCatalog.get("reindeer_val").set(thing_classes=["reindeer"])

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



