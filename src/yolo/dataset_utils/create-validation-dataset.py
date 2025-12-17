import os
import shutil
from random import sample


def main(base_dir):
    train_images_dir = os.path.join(base_dir, "train/images")
    train_labels_dir = os.path.join(base_dir, "train/labels")
    val_images_dir = os.path.join(base_dir, "val/images")
    val_labels_dir = os.path.join(base_dir, "val/labels")

    # Create validation directories if they do not exist
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Gather all image filenames
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".png")]
    total_images = len(image_files)

    # Determine the number of validation images
    val_size = int(0.2 * total_images)

    # Randomly select validation images
    val_images = sample(image_files, val_size)

    # Move selected images and their corresponding label files
    for image in val_images:
        # Move image
        shutil.move(
            os.path.join(train_images_dir, image), os.path.join(val_images_dir, image)
        )

        # Construct label filename
        label_file = image.replace(".png", ".txt")

        # Check if corresponding label file exists, then move
        if os.path.exists(os.path.join(train_labels_dir, label_file)):
            shutil.move(
                os.path.join(train_labels_dir, label_file),
                os.path.join(val_labels_dir, label_file),
            )

    print(f"Moved {val_size} images and their labels to validation directories.")


if __name__ == "__main__":
    base_dir = "/home/taheera.ahmed/data/reindeerdrone-yolo/tiles"

    main(base_dir)
