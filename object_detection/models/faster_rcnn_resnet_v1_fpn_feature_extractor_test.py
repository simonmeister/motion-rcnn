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

"""Tests for object_detection.models.faster_rcnn_resnet_v1_feature_extractor."""

import numpy as np
import tensorflow as tf

from object_detection.models import faster_rcnn_resnet_v1_fpn_feature_extractor as faster_rcnn_resnet_v1_fpn


class FasterRcnnResnetV1FeatureExtractorTest(tf.test.TestCase):

  def _build_feature_extractor(self,
                               architecture='resnet_v1_101'):
    feature_extractor_map = {
        'resnet_v1_50_fpn':
            faster_rcnn_resnet_v1_fpn.FasterRCNNResnet50FPNFeatureExtractor,
    }
    return feature_extractor_map[architecture](
        is_training=False,
        reuse_weights=None,
        first_stage_features_stride=16,
        weight_decay=0.0)

  def test_extract_proposal_features_returns_expected_sizes(self):
    for architecture in ['resnet_v1_50_fpn']:
      feature_extractor = self._build_feature_extractor(
          architecture=architecture)
      preprocessed_inputs = tf.random_uniform(
          [4, 256, 256, 3], maxval=255, dtype=tf.float32)
      rpn_features = feature_extractor.extract_proposal_features(
          preprocessed_inputs, scope='TestScope')

      expected_shapes = [
        (4, 4, 4, 256),
        (4, 8, 8, 256),
        (4, 16, 16, 256),
        (4, 32, 32, 256),
        (4, 64, 64, 256)]

      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        rpn_features_out = sess.run(rpn_features)
        self.assertEqual(len(rpn_features_out), 5)
        for feature_map_out, expected_shape in zip(
            rpn_features_out, expected_shapes):
          shape_out = feature_map_out.shape
          self.assertAllEqual(shape_out, expected_shape)

  def test_extract_box_classifier_features_returns_expected_size(self):
    for architecture in ['resnet_v1_50_fpn']:
      feature_extractor = self._build_feature_extractor(
          architecture=architecture)
      proposal_feature_maps = tf.random_uniform(
          [3, 14, 14, 1024], maxval=255, dtype=tf.float32)
      proposal_classifier_features = (
          feature_extractor.extract_box_classifier_features(
              proposal_feature_maps, scope='TestScope'))
      features_shape = tf.shape(proposal_classifier_features)

      init_op = tf.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(init_op)
        features_shape_out = sess.run(features_shape)
        self.assertAllEqual(features_shape_out, [3, 14, 14, 1024])


if __name__ == '__main__':
  tf.test.main()
