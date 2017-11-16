import tensorflow as tf

from nets import resnet_utils
from nets import resnet_v1


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
