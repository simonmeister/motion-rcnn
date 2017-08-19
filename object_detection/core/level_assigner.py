import tensorflow as tf

# TODO to tensorflow
def assign_to_levels(boxes, im_size, num_levels=6, finest_stride=3):
    """Assigns boxes to pyramid levels.

    Args:
        boxes: array of shape (N, 4), [[x1, y1, x2, y2], ...]
        im_size: [height, width] of full image
        num_levels: number of pyramid levels
        finest_stride: average stride of box side length

    Returns:
        level_ids: array of shape (N,), per-box level indices,
            ranging from coarse to fine
    """
    im_area = im_size[0] * im_size[1]
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    areas = widths * heights
    # if e.g. the finest level has a stride of 4, we want all boxes
    # at 1/4 image resolution to map to the coarsest level (k = 0)
    k = np.round(np.log2(math.sqrt(im_area) / np.sqrt(areas)) - math.log2(finest_stride))
    k = k.astype(np.int32)
    inds = np.where(k < 0)[0]
    k[inds] = 0
    inds = np.where(k > num_levels)[0]
    k[inds] = num_levels
    return k
