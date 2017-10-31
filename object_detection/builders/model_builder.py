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

"""A function to build a DetectionModel from configuration."""
from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_coder_builder
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.core import box_predictor
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.models import faster_rcnn_resnet_v1_fpn_feature_extractor as frcnn_resnet_v1_fpn
from object_detection.protos import model_pb2


# A map of names to Faster R-CNN feature extractors. # TODO extend for FPN
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,

    'faster_rcnn_resnet50_fpn':
    frcnn_resnet_v1_fpn.FasterRCNNResnet50FPNFeatureExtractor,
}


def build(model_config, is_training):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.

  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')
  meta_architecture = model_config.WhichOneof('model')
  if meta_architecture == 'faster_rcnn':
    return _build_faster_rcnn_model(model_config.faster_rcnn, is_training)
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))


def _build_faster_rcnn_feature_extractor(
    feature_extractor_config, is_training, reuse_weights=None):
  """Builds a faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Args:
    feature_extractor_config: A FasterRcnnFeatureExtractor proto config from
      faster_rcnn.proto.
    is_training: True if this feature extractor is being built for training.
    reuse_weights: if the feature extractor should reuse weights.

  Returns:
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  feature_type = feature_extractor_config.type
  first_stage_features_stride = (
      feature_extractor_config.first_stage_features_stride)

  if feature_type not in FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown Faster R-CNN feature_extractor: {}'.format(
        feature_type))
  feature_extractor_class = FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP[
      feature_type]
  return feature_extractor_class(
      is_training, first_stage_features_stride, reuse_weights)


def _build_faster_rcnn_model(frcnn_config, is_training):
  """Builds a Faster R-CNN or R-FCN detection model based on the model config.

  Builds R-FCN model if the second_stage_box_predictor in the config is of type
  `rfcn_box_predictor` else builds a Faster R-CNN model.

  Args:
    frcnn_config: A faster_rcnn.proto object containing the config for the
    desired FasterRCNNMetaArch or RFCNMetaArch.
    is_training: True if this model is being built for training purposes.

  Returns:
    FasterRCNNMetaArch based on the config.
  Raises:
    ValueError: If frcnn_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = frcnn_config.num_classes
  image_resizer_fn = image_resizer_builder.build(frcnn_config.image_resizer)

  feature_extractor = _build_faster_rcnn_feature_extractor(
      frcnn_config.feature_extractor, is_training)

  first_stage_only = frcnn_config.first_stage_only
  first_stage_anchor_generator = anchor_generator_builder.build(
      frcnn_config.first_stage_anchor_generator)

  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate
  first_stage_box_predictor_arg_scope = hyperparams_builder.build(
      frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training)
  first_stage_box_predictor_kernel_size = (
      frcnn_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size
  first_stage_positive_balance_fraction = (
      frcnn_config.first_stage_positive_balance_fraction)
  first_stage_nms_score_threshold = frcnn_config.first_stage_nms_score_threshold
  first_stage_nms_iou_threshold = frcnn_config.first_stage_nms_iou_threshold
  first_stage_max_proposals = frcnn_config.first_stage_max_proposals
  first_stage_loc_loss_weight = (
      frcnn_config.first_stage_localization_loss_weight)
  first_stage_obj_loss_weight = frcnn_config.first_stage_objectness_loss_weight

  initial_crop_size = frcnn_config.initial_crop_size
  maxpool_kernel_size = frcnn_config.maxpool_kernel_size
  maxpool_stride = frcnn_config.maxpool_stride

  second_stage_box_predictor = box_predictor_builder.build(
      hyperparams_builder.build,
      frcnn_config.second_stage_box_predictor,
      is_training=is_training,
      num_classes=num_classes)
  second_stage_batch_size = frcnn_config.second_stage_batch_size
  second_stage_balance_fraction = frcnn_config.second_stage_balance_fraction
  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn
  ) = post_processing_builder.build(frcnn_config.second_stage_post_processing)
  second_stage_localization_loss_weight = (
      frcnn_config.second_stage_localization_loss_weight)
  second_stage_classification_loss_weight = (
      frcnn_config.second_stage_classification_loss_weight)
  second_stage_mask_loss_weight = (
      frcnn_config.second_stage_mask_loss_weight)
  second_stage_motion_loss_weight = (
      frcnn_config.second_stage_motion_loss_weight)
  first_stage_predict_camera_motion = (
      frcnn_config.first_stage_predict_camera_motion)
  first_stage_camera_motion_arg_scope = None
  if first_stage_predict_camera_motion:
    first_stage_camera_motion_arg_scope = hyperparams_builder.build(
        frcnn_config.first_stage_camera_motion_fc_hyperparams, is_training)

  hard_example_miner = None
  if frcnn_config.HasField('hard_example_miner'):
    assert False, "Take care of mask loss" # TODO
    hard_example_miner = losses_builder.build_hard_example_miner(
        frcnn_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  common_kwargs = {
      'is_training': is_training,
      'num_classes': num_classes,
      'image_resizer_fn': image_resizer_fn,
      'feature_extractor': feature_extractor,
      'first_stage_only': first_stage_only,
      'first_stage_anchor_generator': first_stage_anchor_generator,
      'first_stage_atrous_rate': first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope':
      first_stage_box_predictor_arg_scope,
      'first_stage_box_predictor_kernel_size':
      first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth': first_stage_box_predictor_depth,
      'first_stage_minibatch_size': first_stage_minibatch_size,
      'first_stage_positive_balance_fraction':
      first_stage_positive_balance_fraction,
      'first_stage_nms_score_threshold': first_stage_nms_score_threshold,
      'first_stage_nms_iou_threshold': first_stage_nms_iou_threshold,
      'first_stage_max_proposals': first_stage_max_proposals,
      'first_stage_localization_loss_weight': first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight': first_stage_obj_loss_weight,
      'second_stage_batch_size': second_stage_batch_size,
      'second_stage_balance_fraction': second_stage_balance_fraction,
      'second_stage_non_max_suppression_fn':
      second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
      second_stage_localization_loss_weight,
      'second_stage_classification_loss_weight':
      second_stage_classification_loss_weight,
      'second_stage_mask_loss_weight':
      second_stage_mask_loss_weight,
      'second_stage_motion_loss_weight':
      second_stage_motion_loss_weight,
      'hard_example_miner': hard_example_miner,
      'first_stage_predict_camera_motion':
      first_stage_predict_camera_motion,
      'first_stage_camera_motion_loss_weight':
      frcnn_config.second_stage_motion_loss_weight,
      'first_stage_camera_motion_arg_scope':
      first_stage_camera_motion_arg_scope}

  if isinstance(second_stage_box_predictor, box_predictor.RfcnBoxPredictor):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
  else:
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=initial_crop_size,
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
