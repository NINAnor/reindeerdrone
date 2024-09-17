import json
import cv2
import os
import random
import numpy as np
import math
import yaml

from detectron2.structures import BoxMode


def load_custom_annotations(json_path):
    """Load annotations from the new JSON format, removing prefixes from file_upload."""
    with open(json_path, "r") as file:
        data = json.load(file)
    annotations = []
    
    for item in data:
        if "annotations" in item:
            image_annotations = []
            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    value = result["value"]

                    # Check if 'original_width' and 'original_height' are present in result, not in value
                    if "original_width" not in result or "original_height" not in result:
                        # Skip this annotation if these fields are missing
                        print(f"Skipping annotation due to missing dimensions: {result}")
                        continue

                    original_width = result["original_width"]
                    original_height = result["original_height"]

                    # Convert normalized bbox coordinates (x, y, width, height) to pixel values
                    bbox_x = value["x"] * original_width / 100
                    bbox_y = value["y"] * original_height / 100
                    bbox_width = value["width"] * original_width / 100
                    bbox_height = value["height"] * original_height / 100
                    rotation = value.get("rotation", 0)

                    # Get label (we assume first label in rectanglelabels list)
                    category = value["rectanglelabels"][0]

                    # Store the transformed annotation
                    image_annotations.append({
                        "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                        "category": category,
                        "rotation": rotation,
                        "original_width": original_width,
                        "original_height": original_height
                    })
            
            # Remove prefix from the file_upload field
            filename = item["file_upload"].split('-')[-1]  # Split by '-' and take the last part
            
            annotations.append({
                "image_id": item["id"],
                "file_upload": filename,  # Store the cleaned file name without prefix
                "annotations": image_annotations
            })
    
    return annotations

def apply_rotation(bbox, rotation, center):
    """Apply rotation to a bounding box around a center point and return its four corners."""
    if rotation == 0:
        # Return the axis-aligned bounding box corners
        x, y, width, height = bbox
        return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])

    # Convert bbox [x, y, width, height] to corners
    x, y, width, height = bbox
    corners = np.array([
        [x, y],  # top-left
        [x + width, y],  # top-right
        [x + width, y + height],  # bottom-right
        [x, y + height]  # bottom-left
    ])

    # Get rotation matrix for the given angle and center (relative to the tile)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)

    # Apply rotation to each corner
    ones = np.ones(shape=(len(corners), 1))  # Add ones to convert the corners to homogeneous coordinates
    corners_with_ones = np.hstack([corners, ones])
    
    # Apply rotation matrix
    rotated_corners = rotation_matrix.dot(corners_with_ones.T).T

    return rotated_corners

def tile_image_and_adjust_annotations(
    image_path,
    annotations,
    output_dir,
    tile_size=1024,
    overlap=100,
    plot_annotations=False,
):
    """
    Tile an image and adjust annotations for each tile, accounting for rotation and rescaling the bounding boxes.
    """
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    new_annotations = []
    new_images = []
    annotation_id = 1  
    count = 0  

    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            x0, y0 = i, j
            x1, y1 = min(i + tile_size, width), min(j + tile_size, height)
            tile = img[y0:y1, x0:x1]

            pad_width = tile_size - (x1 - x0)
            pad_height = tile_size - (y1 - y0)
            if pad_width > 0 or pad_height > 0:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )

            tile_name = f"{image_id}_{count}.png"
            tile_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(tile_path, tile)

            new_image_entry = {
                "id": f"{image_id}_{count}",
                "file_name": tile_name,
                "width": tile_size,
                "height": tile_size,
            }
            new_images.append(new_image_entry)

            for anno in annotations:
                for bbox in anno["annotations"]:
                    xmin, ymin, box_width, box_height = bbox["bbox"]
                    rotation = bbox.get("rotation", 0)
                    
                    # Compute the bounding box's max coordinates
                    xmax = xmin + box_width
                    ymax = ymin + box_height

                    # Check if the bounding box overlaps with the current tile
                    if xmax > x0 and xmin < x1 and ymax > y0 and ymin < y1:
                        # Shift the bounding box coordinates relative to the tile's origin
                        shifted_xmin = max(xmin, x0) - x0
                        shifted_ymin = max(ymin, y0) - y0
                        shifted_xmax = min(xmax, x1) - x0
                        shifted_ymax = min(ymax, y1) - y0

                        # Recalculate the bounding box width and height within the tile
                        new_bbox_width = max(0, shifted_xmax - shifted_xmin)
                        new_bbox_height = max(0, shifted_ymax - shifted_ymin)

                        if new_bbox_width > 0 and new_bbox_height > 0:
                            new_bbox = [
                                shifted_xmin,
                                shifted_ymin,
                                new_bbox_width,
                                new_bbox_height,
                            ]

                            # Apply rotation to the new bounding box if necessary
                            if rotation != 0:
                                # Calculate the center after shifting to the tile's local coordinates
                                center = (
                                    shifted_xmin + new_bbox_width / 2,
                                    shifted_ymin + new_bbox_height / 2
                                )

                                # Get the rotated corners
                                rotated_bbox = apply_rotation(new_bbox, rotation, center)

                                if plot_annotations:
                                    # Draw the rotated bounding box as a polygon
                                    cv2.polylines(
                                        tile,
                                        [np.int32(rotated_bbox)],
                                        isClosed=True,
                                        color=(0, 255, 0),
                                        thickness=2,
                                    )

                            else:
                                if plot_annotations:
                                    # Draw the axis-aligned bounding box
                                    cv2.rectangle(
                                        tile,
                                        (int(new_bbox[0]), int(new_bbox[1])),
                                        (
                                            int(new_bbox[0] + new_bbox[2]),
                                            int(new_bbox[1] + new_bbox[3]),
                                        ),
                                        (0, 255, 0),
                                        2,
                                    )

                            new_anno = {
                                "id": annotation_id,
                                "bbox": new_bbox,
                                "image_id": new_image_entry["id"],
                                "category_id": bbox["category"],
                                "area": new_bbox_width * new_bbox_height,
                                "iscrowd": 0,
                            }
                            annotation_id += 1
                            new_annotations.append(new_anno)

            count += 1

    return new_images, new_annotations


def clip_rotated_bbox_to_tile(rotated_bbox, tile_x, tile_y, tile_size):
    """
    Clips a rotated bounding box to the boundaries of the tile.
    """
    # Create a polygon for the tile's boundaries
    tile_polygon = np.array([
        [tile_x, tile_y],
        [tile_x + tile_size, tile_y],
        [tile_x + tile_size, tile_y + tile_size],
        [tile_x, tile_y + tile_size]
    ])

    # Create a polygon for the rotated bounding box
    rotated_bbox_polygon = np.array(rotated_bbox)

    # Clip rotated bbox
    clipped_bbox = np.clip(rotated_bbox_polygon, [tile_x, tile_y], [tile_x + tile_size, tile_y + tile_size])

    return clipped_bbox






def process_dataset(
    dataset_dir, annotation_file, output_dir, tile_size, overlap, plot_annotations
):
    """Process the dataset using custom annotations and tile the images."""
    annotations = load_custom_annotations(annotation_file)
    all_new_images = []
    all_new_annotations = []

    for anno in annotations:
        image_path = os.path.join(dataset_dir, anno["file_upload"])
        new_images, new_annotations = tile_image_and_adjust_annotations(
            image_path, [anno], output_dir, tile_size, overlap, plot_annotations
        )
        all_new_images.extend(new_images)
        all_new_annotations.extend(new_annotations)

    # Create the new dataset
    new_data = {
        "images": all_new_images,
        "annotations": all_new_annotations,
    }

    with open(os.path.join(output_dir, "new_annotations.json"), "w") as f:
        json.dump(new_data, f, indent=4)

def split_dataset(img_dir, annotations_file, split_ratio=0.8):
    """Split the dataset into training and validation sets."""
    with open(annotations_file) as f:
        coco = json.load(f)

    image_annotations = {
        img["file_name"].split("/")[-1].split(".")[0]: img["id"]
        for img in coco["images"]
    }
    annotations = {img_id: [] for img_id in image_annotations.values()}

    for anno in coco["annotations"]:
        annotations[anno["image_id"]].append(anno)

    image_files = list(image_annotations.keys())
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    train_annotations = [
        {
            "image_id": image_annotations[file],
            "annotations": annotations[image_annotations[file]],
        }
        for file in train_files
    ]
    val_annotations = [
        {
            "image_id": image_annotations[file],
            "annotations": annotations[image_annotations[file]],
        }
        for file in val_files
    ]

    return train_files, val_files, train_annotations, val_annotations


if __name__ == "__main__":
    # Load config file
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    process_dataset(
        cfg["IMAGES_FOLDER_PATH"],
        cfg["ANNOTATION_PATH"],
        cfg["TILE_FOLDER_PATH"],
        cfg["TILE_SIZE"],
        cfg["OVERLAP"],
        cfg["PLOT_ANNOTATION"],
    )
