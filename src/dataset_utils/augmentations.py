import detectron2.data.transforms as T

def build_augmentation(cfg, is_train=True):
    if is_train:
        augmentation = [
            T.Resize((800, 800)),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomRotation(angle=[-10, 10], expand=False),
            T.RandomCrop("relative", (0.8, 0.8))
        ]
    else:
        augmentation = [
            T.Resize((800, 800))
        ]
    return augmentation
