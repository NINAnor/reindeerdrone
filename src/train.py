#!/usr/bin/env python3

import os
import yaml
import optuna

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
from detectron2.evaluation import COCOEvaluator
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
        #hooks.insert(-1, EarlyStoppingHook(patience=1000, threshold=0.001))
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


def objective(trial, output_dir):
    # Set up the config as before
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Suggest values for hyperparameters using Optuna
    cfg.SOLVER.BASE_LR = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.SOLVER.MAX_ITER = trial.suggest_int("max_iter", 1000, 5000)
    cfg.SOLVER.IMS_PER_BATCH = trial.suggest_categorical("ims_per_batch", [2, 4, 8])
    
    # Configure the rest of the setup
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x")
    cfg.DATASETS.TRAIN = ("reindeer_train",)
    cfg.DATASETS.TEST = ("reindeer_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.TEST.EVAL_PERIOD = 100
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.CUDNN_BENCHMARK = True
    
    # TODO: Set the output directory for the trial using config
    
    output_dir = f"{output_dir}/optuna_trial_{trial.number}"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    params_file = os.path.join(output_dir, 'trial_params.yaml')
    with open(params_file, 'w') as f:
        trial_params = {
            'trial_number': trial.number,
            'lr': cfg.SOLVER.BASE_LR,
            'max_iter': cfg.SOLVER.MAX_ITER,
            'ims_per_batch': cfg.SOLVER.IMS_PER_BATCH,
        }
        yaml.dump(trial_params, f, default_flow_style=False)

    # initialize detectron2 logger
    logger = setup_logger(output=cfg.OUTPUT_DIR)
    logger.info(f"OPTUNA: Starting trial {trial.number} with parameters: LR={cfg.SOLVER.BASE_LR}, MAX_ITER={cfg.SOLVER.MAX_ITER}, IMS_PER_BATCH={cfg.SOLVER.IMS_PER_BATCH}")
    
    # start training
    trainer = ReindeerTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    trainer.train()

    validation_loss = trainer.storage.history("validation_loss").latest()
    
    # log the validation loss for the trial
    logger.info(f"Trial {trial.number} finished with validation loss: {validation_loss}")
    
    return validation_loss

def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '../config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)
    
    img_dir = cfgP["TILE_FOLDER_PATH"]
    annotations_file = cfgP["TILE_ANNOTATION_PATH"]
    optuna_trails = cfgP.get("OPTUNA_TRIALS", 20)
    output_dir = cfgP["OUTPUT_FOLDER"]
    
    # Register the dataset
    classes = ["Adult", "Calf"]
    _, _, train_annotations, val_annotations = split_dataset(annotations_file)
    
    DatasetCatalog.register(
        "reindeer_train", lambda: get_reindeer_dicts(img_dir, train_annotations)
    )
    MetadataCatalog.get("reindeer_train").set(thing_classes=classes)
    
    DatasetCatalog.register(
        "reindeer_val", lambda: get_reindeer_dicts(img_dir, val_annotations)
    )
    MetadataCatalog.get("reindeer_val").set(thing_classes=classes)
    
    # create Optuna study with validation loss
    study = optuna.create_study(direction="minimize")
    
    # optimize the objective function
    study.optimize(lambda trial: objective(trial, output_dir=output_dir), n_trials=optuna_trails)

    # output the best trial
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")


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
