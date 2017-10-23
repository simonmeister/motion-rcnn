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

"""Resnet FPN V1 Faster R-CNN implementation.

See "Feature Pyramid Networks for Object Detection" by Lin et al., 2017.

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""
import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import resnet_utils
from nets import resnet_v1

slim = tf.contrib.slim


# TODO try if this is neccessary and if it works without compromising the pre-training
def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the first unit.
      All other units have stride=1.
      Note that the default slim implementation places the stride in the last unit,
      which is less memory efficient and a deviation from the resnet paper.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, resnet_v1.bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }] + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1))


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v1_50'):
  """Unlike the slim default we use a stride of 2 in the last block."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),]
  return resnet_v1.resnet_v1(
      inputs, blocks, num_classes, is_training,
      global_pool=global_pool, output_stride=output_stride,
      include_root_block=True, spatial_squeeze=spatial_squeeze,
      reuse=reuse, scope=scope)


class FasterRCNNResnetV1FPNFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Resnet V1 feature extractor implementation."""

  def __init__(self,
               architecture,
               resnet_model,
               is_training,
               handles_map,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      architecture: Architecture name of the Resnet V1 model.
      resnet_model: Definition of the Resnet V1 model.
      is_training: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    self._architecture = architecture
    self._resnet_model = resnet_model
    self._handles_map = handles_map
    super(FasterRCNNResnetV1FPNFeatureExtractor, self).__init__(
        is_training, 32, reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    channel_means = [123.68, 116.779, 103.939]
    channel_means = tf.cond(
        tf.equal(tf.shape(resized_inputs)[3], 6),
        lambda: tf.constant(channel_means * 2, dtype=tf.float32),
        lambda: tf.constant(channel_means, dtype=tf.float32))
    return resized_inputs - channel_means # [[channel_means]]

  @property
  def extracted_proposal_features_strides(self):
    return [(x, x) for x in [64, 32, 16, 8, 4]]

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_features: A list of tensors with shape [batch, height, width, depth]
    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      # Disables batchnorm for fine-tuning with smaller batch sizes.
      # TODO: Figure out if it is needed when image batch size is bigger.
      with slim.arg_scope(
          resnet_utils.resnet_arg_scope(
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              weight_decay=self._weight_decay)):
        with tf.variable_scope(
            self._architecture, reuse=self._reuse_weights) as var_scope:
          _, activations = self._resnet_model(
              preprocessed_inputs,
              num_classes=None,
              is_training=False,
              global_pool=False,
              output_stride=None,
              spatial_squeeze=False,
              scope=var_scope)

    return self._build_pyramid(activations, self._handles_map, scope)

  def _build_pyramid(self, end_points, handles_map, scope):
    pyramid = []
    scope_prefix = scope + '/'
    with tf.variable_scope('pyramid'):
        C5 = end_points[scope_prefix + handles_map['C5']]
        P5 = slim.conv2d(C5, 256, [1, 1], stride=1, scope='P5')
        P6 = resnet_utils.subsample(P5, 2)
        pyramid = [P6, P5]

        for c in [4, 3, 2]:
            this_C = end_points[scope_prefix + handles_map['C{}'.format(c)]]
            prev_P = pyramid[-1]

            up_shape = tf.shape(this_C)
            prev_P_up = tf.image.resize_bilinear(
                prev_P,
                [up_shape[1], up_shape[2]],
                name='C{}/upscale'.format(c))

            this_C_adapted = slim.conv2d(this_C, 256, [1,1], stride=1,
                                         scope='C{}'.format(c))

            this_P = tf.add(prev_P_up, this_C_adapted,
                            name='C{}/add'.format(c))
            this_P = slim.conv2d(this_P, 256, [3,3], stride=1,
                                 scope='C{}/refine'.format(c))
            pyramid.append(this_P)
    return pyramid

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    return proposal_feature_maps


class FasterRCNNResnet50FPNFeatureExtractor(FasterRCNNResnetV1FPNFeatureExtractor):
  """Faster R-CNN Resnet 50 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    handles_map = {
        'C2': 'resnet_v1_50/resnet_v1_50/block1/unit_2/bottleneck_v1',
        'C3': 'resnet_v1_50/resnet_v1_50/block2/unit_3/bottleneck_v1',
        'C4': 'resnet_v1_50/resnet_v1_50/block3/unit_5/bottleneck_v1',
        'C5': 'resnet_v1_50/resnet_v1_50/block4/unit_3/bottleneck_v1'}
    super(FasterRCNNResnet50FPNFeatureExtractor, self).__init__(
        'resnet_v1_50', resnet_v1.resnet_v1_50, is_training,
        handles_map, reuse_weights, weight_decay)
