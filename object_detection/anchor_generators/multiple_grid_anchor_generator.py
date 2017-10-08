# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generates grid anchors on the fly corresponding to multiple CNN layers.

Generates grid anchors on the fly corresponding to multiple CNN layers as
described in:
"Feature Pyramid Networks for Object Detection",
Lin et al.
"""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops
from object_detection.core import box_list


class MultipleGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for multiple CNN layers."""

  def __init__(self,
               box_specs_list,
               base_anchor_size=None,
               clip_window=None):
    """Constructs a MultipleGridAnchorGenerator.

    To construct anchors, at multiple grid resolutions, one must provide a
    list of feature_map_shape_list (e.g., [(8, 8), (4, 4)]), and for each grid
    size, a corresponding list of (scale, aspect ratio) box specifications.

    For example:
    box_specs_list = [[(.1, 1.0), (.1, 2.0)],  # for 8x8 grid
                      [(.2, 1.0), (.3, 1.0), (.2, 2.0)]]  # for 4x4 grid

    To support the fully convolutional setting, we pass grid sizes in at
    generation time, while scale and aspect ratios are fixed at construction
    time.

    Args:
      box_specs_list: list of list of (scale, aspect ratio) pairs with the
        outside list having the same number of entries as feature_map_shape_list
        (which is passed in at generation time).
      base_anchor_size: base anchor size as [height, width]
                        (length-2 float tensor, default=[256, 256]).
      clip_window: a tensor of shape [4] specifying a window to which all
        anchors should be clipped. If clip_window is None, then no clipping
        is performed.

    Raises:
      ValueError: if box_specs_list is not a list of list of pairs
      ValueError: if clip_window is not either None or a tensor of shape [4]
    """
    if isinstance(box_specs_list, list) and all(
        [isinstance(list_item, list) for list_item in box_specs_list]):
      self._box_specs = box_specs_list
    else:
      raise ValueError('box_specs_list is expected to be a '
                       'list of lists of pairs')
    if base_anchor_size is None:
      base_anchor_size = tf.constant([256, 256], dtype=tf.float32)
    self._base_anchor_size = base_anchor_size
    if clip_window is not None and clip_window.get_shape().as_list() != [4]:
      raise ValueError('clip_window must either be None or a shape [4] tensor')
    self._clip_window = clip_window
    self._scales = []
    self._aspect_ratios = []
    for box_spec in self._box_specs:
      if not all([isinstance(entry, tuple) and len(entry) == 2
                  for entry in box_spec]):
        raise ValueError('box_specs_list is expected to be a '
                         'list of lists of pairs')
      scales, aspect_ratios = zip(*box_spec)
      self._scales.append(scales)
      self._aspect_ratios.append(aspect_ratios)

  def name_scope(self):
    return 'MultipleGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the Generate function.
    """
    return [len(box_specs) for box_specs in self._box_specs]

  def _generate(self,
                feature_map_shape_list,
                im_height=1,
                im_width=1,
                anchor_strides=None,
                anchor_offsets=None):
    """Generates a collection of bounding boxes to be used as anchors.

    The number of anchors generated for a single grid with shape MxM where we
    place k boxes over each grid center is k*M^2 and thus the total number of
    anchors is the sum over all grids. In our box_specs_list example
    (see the constructor docstring), we would place two boxes over each grid
    point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and
    thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the
    output anchors follows the order of how the grid sizes and box_specs are
    specified (with box_spec index varying the fastest, followed by width
    index, then height index, then grid index).

    Args:
      feature_map_shape_list: list of pairs of convnet layer resolutions in the
        format [(height_0, width_0), (height_1, width_1), ...]. For example,
        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
        correspond to an 8x8 layer followed by a 7x7 layer.
      im_height: the height of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        normalized coordinates, otherwise absolute coordinates are used for the
        grid.
      im_width: the width of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        normalized coordinates, otherwise absolute coordinates are used for the
        grid.
      anchor_strides: list of pairs of strides (in y and x directions
        respectively). For example, setting
        anchor_strides=[(.25, .25), (.5, .5)] means that we want the anchors
        corresponding to the first layer to be strided by .25 and those in the
        second layer to be strided by .5 in both y and x directions. By
        default, if anchor_strides=None, then they are set to be the reciprocal
        of the corresponding grid sizes. The pairs can also be specified as
        dynamic tf.int or tf.float numbers, e.g. for variable shape input
        images.
      anchor_offsets: list of pairs of offsets (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offsets=[(.125, .125), (.25, .25)]) means that we want the
        (0, 0)-th anchor of the first layer to lie at (.125, .125) in image
        space and likewise that we want the (0, 0)-th anchor of the second
        layer to lie at (.25, .25) in image space. By default, if
        anchor_offsets=None, then they are set to be half of the corresponding
        anchor stride. The pairs can also be specified as dynamic tf.int or
        tf.float numbers, e.g. for variable shape input images.

    Returns:
      boxes: a BoxList holding a collection of N anchor boxes
    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == len(self._box_specs)):
      raise ValueError('feature_map_shape_list must be a list with the same '
                       'length as self._box_specs')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')
    if not anchor_strides:
      anchor_strides = [(tf.to_float(im_height) / tf.to_float(pair[0]),
                         tf.to_float(im_width) / tf.to_float(pair[1]))
                        for pair in feature_map_shape_list]
    if not anchor_offsets:
      anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                        for stride in anchor_strides]
    for arg, arg_name in zip([anchor_strides, anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if not all([isinstance(list_item, tuple) and len(list_item) == 2
                  for list_item in arg]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

    anchor_grid_list = []
    min_im_shape = tf.to_float(tf.minimum(im_height, im_width))
    base_anchor_size = min_im_shape * self._base_anchor_size
    for grid_size, scales, aspect_ratios, stride, offset in zip(
        feature_map_shape_list, self._scales, self._aspect_ratios,
        anchor_strides, anchor_offsets):
      anchor_grid_list.append(
          grid_anchor_generator.tile_anchors(
              grid_height=grid_size[0],
              grid_width=grid_size[1],
              scales=scales,
              aspect_ratios=aspect_ratios,
              base_anchor_size=base_anchor_size,
              anchor_stride=stride,
              anchor_offset=offset))
    concatenated_anchors = box_list_ops.concatenate(anchor_grid_list)
    num_anchors = concatenated_anchors.num_boxes_static()
    if num_anchors is None:
      num_anchors = concatenated_anchors.num_boxes()
    if self._clip_window is not None:
      clip_window = tf.multiply(
          tf.to_float([im_height, im_width, im_height, im_width]),
          self._clip_window)
      concatenated_anchors = box_list_ops.clip_to_window(
          concatenated_anchors, clip_window, filter_nonoverlapping=False)
      # TODO: make reshape an option for the clip_to_window op
      concatenated_anchors.set(
          tf.reshape(concatenated_anchors.get(), [num_anchors, 4]))

    stddevs_tensor = 0.01 * tf.ones(
        [num_anchors, 4], dtype=tf.float32, name='stddevs')
    concatenated_anchors.add_field('stddev', stddevs_tensor)

    return concatenated_anchors


class FpnAnchorGenerator(MultipleGridAnchorGenerator):
  def __init__(self, scales,
               aspect_ratios=[0.5, 1.0, 2.0],
               base_anchor_size=None):
    """Constructs a FpnAnchorGenerator. See the paper by Lin et al.

    Anchors that are returned by calling the `generate` method on the returned
    MultipleGridAnchorGenerator object are always in normalized coordinates
    and clipped to the unit square: (i.e. all coordinates lie in [0, 1]x[0, 1]).

    Args:
    num_layers: integer number of grid layers to create anchors for (actual
      grid sizes passed in at generation time)
    aspect_ratios: list or tuple of (float) aspect ratios to place on each
      pyramid location.
    base_anchor_size: base anchor size as [height, width].
    """
    self._pyramid_scales = scales
    box_specs_list = [[(scale, aspect_ratio)
        for aspect_ratio in aspect_ratios] for scale in scales]
    super().__init__(box_specs_list, base_anchor_size)

  def assign_boxes_to_layers(self, absolute_boxes):
    """Assigns boxes to feature pyramid layers.

    Args:
      absolute_boxes: A float32 tensor with shape [batch_size,
        num_proposals, box_code_size] containing boxes in
        absolute coordinates.

    Returns:
      layer_indices: A int32 tensor of shape [N],
        ranging from lowest resolution (0) to highest (num_resolutions - 1)
    """
    def log2(x):
      numerator = tf.log(x)
      denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
      return numerator / denominator

    min_scale = self._base_anchor_size[0] * min(self._pyramid_scales)
    box_scales = tf.sqrt(box_list_ops.area(box_list.BoxList(absolute_boxes)))

    max_k = len(self._pyramid_scales) - 1
    k = tf.round(log2(box_scales / min_scale))
    k = tf.minimum(tf.maximum(k, 0), max_k)

    return tf.to_int32(k)


if __name__ == '__main__':
  #scales = [0.125, 0.25, 0.5, 1.0, 2.0]
  scales = [2.0, 1.0, 0.5, 0.25, 0.125]
  anchor_generator = FpnAnchorGenerator(
      scales=scales,
      aspect_ratios=[0.5, 1.0, 2.0],
      base_anchor_size=(256, 256))
  feature_map_shape_list = [(16, 32), (32, 64), (64, 128),
                            (128, 256), (256, 512)]
  anchors_boxlist = anchor_generator.generate(
      feature_map_shape_list=feature_map_shape_list,
      anchor_strides=[(x, x) for x in [64, 32, 16, 8, 4]])

  clip_window = tf.to_float(tf.stack([0, 0, 1024, 2048]))
  #anchors_boxlist, keep_indices = box_list_ops.prune_outside_window(
  #    anchors_boxlist, clip_window)

  anchors = anchors_boxlist.get()

  with tf.Session():
    print(anchors[0:100].eval())
