import json
import yaml
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

import pytorch_lightning as pl


from dataset import get_reindeer_dicts, split_dataset

from yaml import FullLoader

class TrainerModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.trainer = DefaultTrainer(cfg)
        self.trainer.resume_or_load(resume=False)

    def train_dataloader(self):
        return self.trainer.data_loader

    def training_step(self, batch, batch_idx):
        loss_dict = self.trainer.model(batch)
        losses = sum(loss_dict.values())
        self.log("train_loss", losses)
        return losses

    def configure_optimizers(self):
        return self.trainer.optimizer


if __name__ == "__main__":

    with open("./config.yaml") as f:
        cfgP = yaml.load(f, Loader=FullLoader)

    # Load configuration from file
    cfg = get_cfg()
    cfg.merge_from_file("detectron2.yaml")

    # Split the dataset
    img_dir = cfgP["TILE_FOLDER_PATH"]
    annotations_file = cfgP["TILE_ANNOTATION_PATH"]
    train_files, val_files, train_annotations, val_annotations = split_dataset(img_dir, annotations_file)

    # Save split annotations for reproducibility (optional)
    with open("train_annotations.json", "w") as f:
        json.dump(train_annotations, f, indent=4)
    with open("val_annotations.json", "w") as f:
        json.dump(val_annotations, f, indent=4)

    # Register the dataset
    DatasetCatalog.register("reindeer_train", lambda: get_reindeer_dicts(img_dir, train_annotations))
    MetadataCatalog.get("reindeer_train").set(thing_classes=["reindeer"])

    DatasetCatalog.register("reindeer_val", lambda: get_reindeer_dicts(img_dir, val_annotations))
    MetadataCatalog.get("reindeer_val").set(thing_classes=["reindeer"])

    #trainer_module = TrainerModule(cfg)
    #trainer = pl.Trainer(max_epochs=10, gpus=1)
    #trainer.fit(trainer_module)
