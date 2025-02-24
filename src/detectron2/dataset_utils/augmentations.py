import detectron2.data.transforms as T

def build_augmentation(cfg, is_train=True):
    if is_train:
        augmentation = [
            T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
            T.RandomFlip(),
            T.RandomLighting(scale=0.9),
        ]
    else:
        augmentation = [ 
            T.ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')
        ]
    return augmentation
