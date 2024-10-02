import logging
import os
import yaml
import json

from yaml import FullLoader
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import DefaultTrainer, launch, default_argument_parser
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from dataset_utils import get_reindeer_dicts, create_test_dataset
from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter
from detectron2.engine import DefaultTrainer


def setup(args, output_dir):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Ensure you adjust this to the number of your classes

    cfg.DATASETS.TEST = ("reindeer_test",)
    cfg.DATASETS.TRAIN = ("reindeer_train",)
    
    cfg.TEST.EVAL_PERIOD = 10
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    
    eval_log_dir = os.path.join(cfg.OUTPUT_DIR, "evaluation")
    os.makedirs(eval_log_dir, exist_ok=True)
    
    logger = setup_logger(output=eval_log_dir)
    logger.setLevel(logging.INFO)

    return cfg


def evaluate(args):
    setup_logger()
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, './../config.yaml')
    
    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)
        
    test_img_dir = cfgP["TILE_TEST_FOLDER_PATH"]
    test_anno_file  = cfgP["TILE_TEST_ANNOTATION_PATH"]
    train_img_dir = cfgP["TILE_FOLDER_PATH"]
    train_anno_file = cfgP["TILE_ANNOTATION_PATH"]
    output_dir = cfgP["OUTPUT_FOLDER"]
    
    cfg = setup(args, output_dir=output_dir)
    
    
    classes = ["Adult", "Calf"]
    
    # for some reason it needs the train dataset to be registered before the test dataset
    _, train_anno = create_test_dataset(train_anno_file)
        
    DatasetCatalog.register(
        "reindeer_train", lambda: get_reindeer_dicts(train_img_dir, train_anno)
    )
    MetadataCatalog.get("reindeer_train").set(thing_classes=classes)
    
   
    # register test data
    _, test_anno = create_test_dataset(test_anno_file)
    DatasetCatalog.register(
        "reindeer_test", lambda: get_reindeer_dicts(test_img_dir, test_anno)
    )
    MetadataCatalog.get("reindeer_test").set(thing_classes=classes)
    
    # perform evaluation on the test dataset
    evaluator_test = COCOEvaluator("reindeer_test", cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, "reindeer_test")

    # Use the custom trainer to load the model and perform inference
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    inference_on_dataset(trainer.model, test_loader, evaluator_test)

    # Close the TensorBoard writer after evaluation
    trainer.writer.close()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        evaluate,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()