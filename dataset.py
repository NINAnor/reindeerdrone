import json
import yaml
import cv2
import os
from yaml import FullLoader
from detectron2.structures import BoxMode

# Load COCO annotations
def load_coco_annotations(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data

# Tile the image with overlap
def tile_image(image, tile_size=1024, overlap=100):
    tiles = []
    height, width, _ = image.shape
    for i in range(0, width, tile_size - overlap):
        for j in range(0, height, tile_size - overlap):
            x1 = min(i + tile_size, width)
            y1 = min(j + tile_size, height)
            tile = image[j:y1, i:x1]
            pad_width = tile_size - (x1 - i)
            pad_height = tile_size - (y1 - j)
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
            tiles.append((tile, i, j))
    return tiles, width, height

# Adjust the annotations to match the tiles
def adjust_annotations_for_tiles(annotations, x0, y0, tile_name, tile_size):
    new_annotations = []
    x1 = x0 + tile_size
    y1 = y0 + tile_size
    annotation_id = 1  # Unique annotation ID counter

    for anno in annotations:
        bbox = anno["bbox"]
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
                new_bbox = [
                    clipped_xmin,
                    clipped_ymin,
                    new_bbox_width,
                    new_bbox_height,
                ]
                new_anno = {
                    "id": annotation_id,
                    "bbox": new_bbox,
                    "image_id": tile_name,
                    "category_id": anno["category_id"],
                    "area": new_bbox_width * new_bbox_height,
                    "iscrowd": 0
                }
                annotation_id += 1
                new_annotations.append(new_anno)

    return new_annotations

# Tile the image and adjust annotations
def tile_image_and_adjust_annotations(image_path, image_info, annotations, output_dir, tile_size, overlap, plot_annotations=False):

    img = cv2.imread(image_path)
    tiles, width, height = tile_image(img, tile_size, overlap)

    # Extract the original file name without extension
    original_file_name = os.path.splitext(os.path.basename(image_path))[0]
    # Filter annotations to only include those relevant to the current image
    image_annotations = [anno for anno in annotations if anno["image_id"] == image_info["id"]]
    new_images = []
    all_new_annotations = []
    
    os.makedirs(output_dir, exist_ok=True)

    for idx, (tile, x0, y0) in enumerate(tiles):
        # Adjust annotations for the current tile
        tile_name = f"{original_file_name}_tile{idx}.png"
        tile_path = os.path.join(output_dir, tile_name)
        new_annotations = adjust_annotations_for_tiles(image_annotations, x0, y0, tile_name, tile_size)

        new_images.append({
            "id": f"{original_file_name}_tile{idx}",
            "file_name": tile_name,
            "width": tile_size,
            "height": tile_size
        })
        
        all_new_annotations.extend(new_annotations)

        if plot_annotations:
            for anno in new_annotations:
                bbox = anno["bbox"]
                print(bbox)
                cv2.rectangle(
                    tile,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                    (0, 255, 0),
                    2,
                )

        # Save the tile
        cv2.imwrite(tile_path, tile)

    return new_images, all_new_annotations

# Process the dataset and create tiles with adjusted annotations
def process_dataset(dataset_dir, annotation_file, output_dir, tile_size, overlap, plot_annotations):
    coco_data = load_coco_annotations(annotation_file)
    annotations = coco_data["annotations"]
    all_new_images = []
    all_new_annotations = []

    for image_info in coco_data["images"]:
        image_path = os.path.join(dataset_dir, image_info["file_name"])

        # Get new images and annotations while preserving original file names
        new_images, new_annotations = tile_image_and_adjust_annotations(
            image_path, image_info, annotations, output_dir, tile_size, overlap, plot_annotations
        )
        all_new_images.extend(new_images)
        all_new_annotations.extend(new_annotations)

    # Create the new COCO-compliant dataset
    new_coco_data = {
        "images": all_new_images,
        "annotations": all_new_annotations,
        "categories": coco_data["categories"],
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", [])
    }

    with open(os.path.join(output_dir, "new_annotations.json"), "w") as f:
        json.dump(new_coco_data, f, indent=4)


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=FullLoader)

    process_dataset(
        cfg["IMAGES_FOLDER_PATH"],
        cfg["ANNOTATION_PATH"],
        cfg["TILE_FOLDER_PATH"],
        cfg["TILE_SIZE"],
        cfg["OVERLAP"],
        cfg["PLOT_ANNOTATION"],
    )
