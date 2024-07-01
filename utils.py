import cv2
import numpy as np


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


def combine_tiles(image, tiles, tile_size=1024, overlap=100):
    height, width, _ = image.shape
    full_image = np.zeros((height, width, 3), dtype=np.uint8)
    for tile, x, y in tiles:
        x1 = min(x + tile_size, width)
        y1 = min(y + tile_size, height)
        pad_width = tile_size - (x1 - x)
        pad_height = tile_size - (y1 - y)
        tile = tile[: tile_size - pad_height, : tile_size - pad_width]
        full_image[y:y1, x:x1] = tile
    return full_image
