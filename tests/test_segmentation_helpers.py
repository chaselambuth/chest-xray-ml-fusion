import numpy as np
from PIL import Image

from xray_fusion.segmentation import cropped_frontal_image, largest_bbox_from_mask, masked_image


# Mask geometry should return PIL-style crop coordinates.
def test_largest_bbox_from_mask_returns_crop_box():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:5] = 1

    assert largest_bbox_from_mask(mask) == (2, 1, 5, 3)


# Empty masks should leave the original image shape available.
def test_cropped_frontal_image_falls_back_when_mask_is_empty():
    image = Image.new("RGB", (5, 4), color=(10, 20, 30))
    mask = np.zeros((4, 5), dtype=np.uint8)

    assert cropped_frontal_image(image, mask).size == (5, 4)


# Binary masking should zero pixels outside the predicted lung region.
def test_masked_image_zeroes_background_pixels():
    image = Image.new("RGB", (2, 2), color=(100, 50, 25))
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    masked = np.array(masked_image(image, mask))

    assert masked[0, 0].tolist() == [100, 50, 25]
    assert masked[0, 1].tolist() == [0, 0, 0]
