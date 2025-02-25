import argparse
import json
from pathlib import Path


def convert_anno_to_yolo(path_to_coco_anno_file, output_folder):
    """
    Convert COCO annotations to YOLO

    Args:
        data (): _description_
        output_folder (_type_): _description_
    """
    with open(path_to_coco_anno_file) as f:
        data = json.load(f)

    image_dimensions = {
        image["id"]: (image["width"], image["height"]) for image in data["images"]
    }

    # process each annotation
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        x_min, y_min, width, height = annotation["bbox"]

        # get dimensions of the image
        img_width, img_height = image_dimensions[image_id.split(".")[0]]

        # convert COCO bbox to YOLO format
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # YOLO format: class x_center y_center width height
        yolo_format = (
            f"{category_id} {x_center} {y_center} {norm_width} {norm_height}\n"
        )

        file_path = output_folder / f"{image_id.split('.')[0]}.txt"
        with open(file_path, "a") as file:
            file.write(yolo_format)


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to YOLO format."
    )
    parser.add_argument(
        "coco_anno_file_path", type=str, help="Path to the COCO annotations JSON file"
    )
    parser.add_argument(
        "output_folder", type=str, help="Output folder for YOLO formatted labels"
    )

    args = parser.parse_args()

    path_to_coco_anno_file = Path(args.coco_anno_file_path)
    output_folder = Path(args.output_folder)

    convert_anno_to_yolo(path_to_coco_anno_file, output_folder)


if __name__ == "__main__":
    main()
