import json
import random
import os

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

def create_test_dataset(annotations_file):
    with open(annotations_file) as f:
        coco = json.load(f)

    image_annotations = {normalize_image_id(img['file_name']): img['id'] for img in coco['images']}
    annotations = {img_id: [] for img_id in image_annotations.values()}

    for anno in coco['annotations']:
        norm_image_id = normalize_image_id(anno['image_id'])
        if norm_image_id in annotations:
            annotations[norm_image_id].append(anno)

    test_files = list(image_annotations.keys())
    test_annotations = [{'image_id': image_annotations[file], 'annotations': annotations[image_annotations[file]]} for file in test_files]

    return test_files, test_annotations