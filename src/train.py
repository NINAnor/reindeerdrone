import logging
import os
import yaml

from yaml import FullLoader
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, launch, default_argument_parser
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from dataset_utils import split_dataset, get_reindeer_dicts, build_augmentation
from hooks import LossEvalHook, EarlyStoppingHook


class ReindeerTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg,
            mapper=DatasetMapper(cfg, is_train=True, augmentations=build_augmentation(cfg, is_train=True)))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        hooks = super().build_hooks()  # Correct super() call
        hooks.insert(-1, EarlyStoppingHook(patience=1000, threshold=0.001))
        hooks.insert(-1, LossEvalHook(
            eval_period = self.cfg.TEST.EVAL_PERIOD,
            model = self.model,
            data_loader = build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            ),
        ))
        return hooks



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
    cfg.SOLVER.MAX_ITER = 3000 # TODO: Change this to 3000
    cfg.TEST.EVAL_PERIOD = 100
    cfg.CUDNN_BENCHMARK = True

    # Assuming you have two classes: "Adult" and "Calf"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Set to 2 classes if you have both "Adult" and "Calf"
    cfg.OUTPUT_DIR = "./../output"

    logger = setup_logger(output=cfg.OUTPUT_DIR)
    logger.setLevel(logging.INFO)
    
    return cfg

def main(args):
    setup_logger()
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open("./../config.yaml") as f:
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
