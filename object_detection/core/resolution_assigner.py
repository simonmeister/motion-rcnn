import tensorflow as tf


def assign_to_resolutions(boxes, im_size, num_resolutions=6, finest_stride=3):
    """Assigns boxes to feature map resolutions.

    Args:
      boxes: array of shape [N, 4], [[y1, x1, y2, x2], ...]
      image_shape: A 1D int32 tensors of size [4] containing the image shape.
      num_resolutions: number of feature maps at different resolutions
      finest_stride: average stride of box side length

    Returns:
      resolution_indices: array of shape [N],
        ranging from lowest resolution (0) to highest (num_resolutions - 1)
    """
    # im_area = im_size[0] * im_size[1]
    im_shortest_side = tf.minimum(image_shape[1], image_shape[2])
    # widths = boxes[:, 2] - boxes[:, 0] + 1.0
    # heights = boxes[:, 3] - boxes[:, 1] + 1.0
    # areas = widths * heights
    # # if e.g. the finest level has a stride of 4, we want all boxes
    # # at 1/4 image resolution to map to the coarsest level (k = 0)
    # k = np.round(np.log2(math.sqrt(im_area) / np.sqrt(areas)) - math.log2(finest_stride))
    heights = boxes[:, 2] - boxes[:, 0]
    widths = boxes[:, 3] - boxes[:, 1]
    areas = heights * widths
    k = tf.floor()
    k = tf.minimum(tf.maximum(k, 0), num_resolutions - 1)
    return k
