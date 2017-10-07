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

"""Tests for object_detection.models.model_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.models import faster_rcnn_resnet_v1_fpn_feature_extractor as frcnn_resnet_v1_fpn
from object_detection.protos import model_pb2

FEATURE_EXTRACTOR_MAPS = {
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,

    'faster_rcnn_resnet50_fpn':
    frcnn_resnet_v1_fpn.FasterRCNNResnet50FPNFeatureExtractor,
}


class ModelBuilderTest(tf.test.TestCase):

  def create_model(self, model_config):
    """Builds a DetectionModel based on the model config.

    Args:
      model_config: A model.proto object containing the config for the desired
        DetectionModel.

    Returns:
      DetectionModel based on the config.
    """
    return model_builder.build(model_config, is_training=True)

def test_create_faster_rcnn_resnet_v1_models_from_config(self):
    model_text_proto = """
      faster_rcnn {
        num_classes: 3
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 600
            max_dimension: 1024
          }
        }
        feature_extractor {
          type: 'faster_rcnn_resnet101'
        }
        first_stage_anchor_generator {
          grid_anchor_generator {
            scales: [0.25, 0.5, 1.0, 2.0]
            aspect_ratios: [0.5, 1.0, 2.0]
            height_stride: 16
            width_stride: 16
          }
        }
        first_stage_box_predictor_conv_hyperparams {
          regularizer {
            l2_regularizer {
            }
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
        initial_crop_size: 14
        maxpool_kernel_size: 2
        maxpool_stride: 2
        second_stage_box_predictor {
          mask_rcnn_box_predictor {
            fc_hyperparams {
              op: FC
              regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
            }
          }
        }
        second_stage_post_processing {
          batch_non_max_suppression {
            score_threshold: 0.01
            iou_threshold: 0.6
            max_detections_per_class: 100
            max_total_detections: 300
          }
          score_converter: SOFTMAX
        }
      }"""
    model_proto = model_pb2.DetectionModel()
    text_format.Merge(model_text_proto, model_proto)
    for extractor_type, extractor_class in FEATURE_EXTRACTOR_MAPS.items():
      model_proto.faster_rcnn.feature_extractor.type = extractor_type
      model = model_builder.build(model_proto, is_training=True)
      self.assertIsInstance(model, faster_rcnn_meta_arch.FasterRCNNMetaArch)
      self.assertIsInstance(model._feature_extractor, extractor_class)

  def test_create_faster_rcnn_model_from_config_with_example_miner(self):
    model_text_proto = """
      faster_rcnn {
        num_classes: 3
        feature_extractor {
          type: 'faster_rcnn_resnet50_fpn'
        }
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 600
            max_dimension: 1024
          }
        }
        first_stage_anchor_generator {
          grid_anchor_generator {
            scales: [0.25, 0.5, 1.0, 2.0]
            aspect_ratios: [0.5, 1.0, 2.0]
            height_stride: 16
            width_stride: 16
          }
        }
        first_stage_box_predictor_conv_hyperparams {
          regularizer {
            l2_regularizer {
            }
          }
          initializer {
            truncated_normal_initializer {
            }
          }
        }
        second_stage_box_predictor {
          mask_rcnn_box_predictor {
            fc_hyperparams {
              op: FC
              regularizer {
                l2_regularizer {
                }
              }
              initializer {
                truncated_normal_initializer {
                }
              }
            }
          }
        }
        hard_example_miner {
          num_hard_examples: 10
          iou_threshold: 0.99
        }
      }"""
    model_proto = model_pb2.DetectionModel()
    text_format.Merge(model_text_proto, model_proto)
    model = model_builder.build(model_proto, is_training=True)
    self.assertIsNotNone(model._hard_example_miner)


if __name__ == '__main__':
  tf.test.main()
