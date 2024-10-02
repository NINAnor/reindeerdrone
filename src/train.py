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
        hooks = super().build_hooks()
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



def setup(args, output_dir):
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
    cfg.TEST.EVAL_PERIOD = 100
    cfg.CUDNN_BENCHMARK = True

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = output_dir

    logger = setup_logger(output=cfg.OUTPUT_DIR)
    logger.setLevel(logging.INFO)
    
    return cfg

def main(args):
    # load the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, './../config.yaml')
    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)
        
    output_dir = cfgP["OUTPUT_FOLDER"]
    img_dir = cfgP["TILE_FOLDER_PATH"]
    annotations_file = cfgP["TILE_ANNOTATION_PATH"]
        
    setup_logger(output_dir)
    cfg = setup(args, output_dir=output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    classes = ["Adult", "Calf"]

    # split the dataset
    _, _, train_annotations, val_annotations = split_dataset(
        annotations_file
    )
    
    # register the dataset
    DatasetCatalog.register(
        "reindeer_train", lambda: get_reindeer_dicts(img_dir, train_annotations)
    )
    MetadataCatalog.get("reindeer_train").set(thing_classes=classes)

    DatasetCatalog.register(
        "reindeer_val", lambda: get_reindeer_dicts(img_dir, val_annotations)
    )
    MetadataCatalog.get("reindeer_val").set(thing_classes=classes)
    
    # start training
    trainer = ReindeerTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # run validation
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
