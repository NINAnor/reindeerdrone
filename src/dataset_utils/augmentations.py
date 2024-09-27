import detectron2.data.transforms as T

def build_augmentation(cfg, is_train=True):
    if is_train:
        augmentation = [
            #T.Resize((800, 800)),  # Resize images to a fixed size
            #T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # Horizontal flip
            #T.RandomRotation(angle=[-10, 10], expand=False),  # Rotate between -10 and 10 degrees
            #T.RandomCrop("relative", (0.8, 0.8))  # Crop randomly to 80% of the image
        ]
    else:
        augmentation = [
            #T.Resize((800, 800))
        ]
    return augmentation
