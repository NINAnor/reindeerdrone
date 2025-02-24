import os
import cv2
from detectron2.structures import BoxMode

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
