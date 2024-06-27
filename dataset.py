import json
import yaml
from yaml import FullLoader

import cv2

import hashlib

import os

def load_coco_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def tile_image_and_adjust_annotations(image_path: str, annotations: list, output_dir: str, tile_size: int, overlap: int, plot_annotations: bool):
    """
    Tile an image and adjust annotations for each tile, drawing the bounding boxes on the tiles and saving them.

    Args:
        image_path (str): Path to the image to be tiled.
        annotations (list): List of annotations dicts, each with 'bbox' and 'image_id'.
        output_dir (str): Directory where the tiled images will be saved.
        tile_size (int): The size of each tile (default is 1024).
        overlap (int): Overlap size between tiles (default is 100).

    Returns:
        dict: Updated annotations for each tile.
    """
    # Read the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # Prepare directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare to collect new annotations
    new_annotations = []
    count = 0  # Counter for tile IDs

    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            # Define the coordinates of the tile
            x0, y0 = i, j
            x1, y1 = min(i + tile_size, width), min(j + tile_size, height)
            tile = img[y0:y1, x0:x1]

            # Check if padding is needed
            pad_width = tile_size - (x1 - x0)
            pad_height = tile_size - (y1 - y0)
            if pad_width > 0 or pad_height > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            draw_tile = tile.copy()  # Copy of the tile for drawing

            # Adjust annotations for the tile
            for anno in annotations:
                bbox = anno['bbox']
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

                # Check if the bounding box intersects the tile
                if xmax > x0 and xmin < x1 and ymax > y0 and ymin < y1:
                    # Clip the bounding box coordinates to fit within the tile
                    clipped_xmin = max(xmin, x0) - x0
                    clipped_ymin = max(ymin, y0) - y0
                    clipped_xmax = min(xmax, x1) - x0
                    clipped_ymax = min(ymax, y1) - y0
                    new_bbox_width = max(0, clipped_xmax - clipped_xmin)
                    new_bbox_height = max(0, clipped_ymax - clipped_ymin)

                    # Ensure we have a valid rectangle
                    if new_bbox_width > 0 and new_bbox_height > 0:
                        new_bbox = [clipped_xmin, clipped_ymin, new_bbox_width, new_bbox_height]
                        new_anno = {
                            'bbox': new_bbox,
                            'image_id': f"{image_id}_{count}",
                            'category_id': anno['category_id']
                        }
                        new_annotations.append(new_anno)

                        # Draw rectangle on the tile
                        if plot_annotations:
                            cv2.rectangle(draw_tile, (int(clipped_xmin), int(clipped_ymin)), 
                                        (int(clipped_xmin + new_bbox_width), int(clipped_ymin + new_bbox_height)), 
                                        (0, 255, 0), 2) 

            # Save the tile with annotations drawn on it
            tile_name = f"{image_id}_{count}.png"
            tile_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(tile_path, draw_tile)

            count += 1

    return new_annotations

def process_dataset(dataset_dir, annotation_file, output_dir, tile_size, overlap, plot_annotations):
    coco_data = load_coco_annotations(annotation_file)
    annotations = coco_data['annotations']

    for image_info in coco_data['images']:
        image_path = os.path.join(dataset_dir, image_info['file_name'])
        new_annotations = tile_image_and_adjust_annotations(image_path, annotations, output_dir, tile_size, overlap, plot_annotations)

    # Assume we collect all new_annotations in a list and save them
    new_coco_data = coco_data  # Copy original structure
    new_coco_data['annotations'] = new_annotations  # Replace with new annotations
    with open(os.path.join(output_dir, 'new_annotations.json'), 'w') as f:
        json.dump(new_coco_data, f)

if __name__ == "__main__":

    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=FullLoader)
    
    process_dataset(cfg["IMAGES_FOLDER_PATH"], 
                    cfg["ANNOTATION_PATH"], 
                    cfg["TILE_FOLDER_PATH"],
                    cfg["TILE_SIZE"],
                    cfg["OVERLAP"],
                    cfg["PLOT_ANNOTATION"])