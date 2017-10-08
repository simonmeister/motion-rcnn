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

"""Tests for anchor_generators.multiple_grid_anchor_generator_test.py."""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import multiple_grid_anchor_generator as ag
from object_detection.core import box_list_ops


class MultipleGridAnchorGeneratorTest(tf.test.TestCase):

  def test_construct_single_anchor_grid(self):
    """Builds a 1x1 anchor grid to test the size of the output boxes."""
    exp_anchor_corners = [[-121, -35, 135, 29], [-249, -67, 263, 61],
                          [-505, -131, 519, 125], [-57, -67, 71, 61],
                          [-121, -131, 135, 125], [-249, -259, 263, 253],
                          [-25, -131, 39, 125], [-57, -259, 71, 253],
                          [-121, -515, 135, 509]]

    base_anchor_size = tf.constant([256, 256], dtype=tf.float32)
    box_specs_list = [[(.5, .25), (1.0, .25), (2.0, .25),
                       (.5, 1.0), (1.0, 1.0), (2.0, 1.0),
                       (.5, 4.0), (1.0, 4.0), (2.0, 4.0)]]
    anchor_generator = ag.MultipleGridAnchorGenerator(
        box_specs_list, base_anchor_size)
    anchors = anchor_generator.generate(feature_map_shape_list=[(1, 1)],
                                        anchor_strides=[(16, 16)],
                                        anchor_offsets=[(7, -3)])
    anchor_corners = anchors.get()
    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid(self):
    base_anchor_size = tf.constant([10, 10], dtype=tf.float32)
    box_specs_list = [[(0.5, 1.0), (1.0, 1.0), (2.0, 1.0)]]

    exp_anchor_corners = [[-2.5, -2.5, 2.5, 2.5], [-5., -5., 5., 5.],
                          [-10., -10., 10., 10.], [-2.5, 16.5, 2.5, 21.5],
                          [-5., 14., 5, 24], [-10., 9., 10, 29],
                          [16.5, -2.5, 21.5, 2.5], [14., -5., 24, 5],
                          [9., -10., 29, 10], [16.5, 16.5, 21.5, 21.5],
                          [14., 14., 24, 24], [9., 9., 29, 29]]

    anchor_generator = ag.MultipleGridAnchorGenerator(
        box_specs_list, base_anchor_size)
    anchors = anchor_generator.generate(feature_map_shape_list=[(2, 2)],
                                        anchor_strides=[(19, 19)],
                                        anchor_offsets=[(0, 0)])
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid_non_square(self):
    base_anchor_size = tf.constant([1, 1], dtype=tf.float32)
    box_specs_list = [[(1.0, 1.0)]]

    exp_anchor_corners = [[0., -0.25, 1., 0.75], [0., 0.25, 1., 1.25]]

    anchor_generator = ag.MultipleGridAnchorGenerator(box_specs_list,
                                                      base_anchor_size)
    anchors = anchor_generator.generate(feature_map_shape_list=[(tf.constant(
        1, dtype=tf.int32), tf.constant(2, dtype=tf.int32))])
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid_unnormalized(self):
    base_anchor_size = tf.constant([1, 1], dtype=tf.float32)
    box_specs_list = [[(1.0, 1.0)]]

    exp_anchor_corners = [[0., 0., 320., 320.], [0., 320., 320., 640.]]

    anchor_generator = ag.MultipleGridAnchorGenerator(box_specs_list,
                                                      base_anchor_size)
    anchors = anchor_generator.generate(
        feature_map_shape_list=[(tf.constant(1, dtype=tf.int32), tf.constant(
            2, dtype=tf.int32))],
        im_height=320,
        im_width=640)
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_multiple_grids(self):
    base_anchor_size = tf.constant([1.0, 1.0], dtype=tf.float32)
    box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                      [(1.0, 1.0), (1.0, 0.5)]]

    # height and width of box with .5 aspect ratio
    h = np.sqrt(2)
    w = 1.0/np.sqrt(2)
    exp_small_grid_corners = [[-.25, -.25, .75, .75],
                              [.25-.5*h, .25-.5*w, .25+.5*h, .25+.5*w],
                              [-.25, .25, .75, 1.25],
                              [.25-.5*h, .75-.5*w, .25+.5*h, .75+.5*w],
                              [.25, -.25, 1.25, .75],
                              [.75-.5*h, .25-.5*w, .75+.5*h, .25+.5*w],
                              [.25, .25, 1.25, 1.25],
                              [.75-.5*h, .75-.5*w, .75+.5*h, .75+.5*w]]
    # only test first entry of larger set of anchors
    exp_big_grid_corners = [[.125-.5, .125-.5, .125+.5, .125+.5],
                            [.125-1.0, .125-1.0, .125+1.0, .125+1.0],
                            [.125-.5*h, .125-.5*w, .125+.5*h, .125+.5*w],]

    anchor_generator = ag.MultipleGridAnchorGenerator(
        box_specs_list, base_anchor_size)
    anchors = anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                        anchor_strides=[(.25, .25), (.5, .5)],
                                        anchor_offsets=[(.125, .125),
                                                        (.25, .25)])
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertEquals(anchor_corners_out.shape, (56, 4))
      big_grid_corners = anchor_corners_out[0:3, :]
      small_grid_corners = anchor_corners_out[48:, :]
      self.assertAllClose(small_grid_corners, exp_small_grid_corners)
      self.assertAllClose(big_grid_corners, exp_big_grid_corners)

  def test_construct_multiple_grids_with_clipping(self):
    base_anchor_size = tf.constant([1.0, 1.0], dtype=tf.float32)
    box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                      [(1.0, 1.0), (1.0, 0.5)]]

    # height and width of box with .5 aspect ratio
    h = np.sqrt(2)
    w = 1.0/np.sqrt(2)
    exp_small_grid_corners = [[0, 0, .75, .75],
                              [0, 0, .25+.5*h, .25+.5*w],
                              [0, .25, .75, 1],
                              [0, .75-.5*w, .25+.5*h, 1],
                              [.25, 0, 1, .75],
                              [.75-.5*h, 0, 1, .25+.5*w],
                              [.25, .25, 1, 1],
                              [.75-.5*h, .75-.5*w, 1, 1]]

    clip_window = tf.constant([0, 0, 1, 1], dtype=tf.float32)
    anchor_generator = ag.MultipleGridAnchorGenerator(
        box_specs_list, base_anchor_size, clip_window=clip_window)
    anchors = anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)])
    anchor_corners = anchors.get()

    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      small_grid_corners = anchor_corners_out[48:, :]
      self.assertAllClose(small_grid_corners, exp_small_grid_corners)

  def test_invalid_box_specs(self):
    # not all box specs are pairs
    box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                      [(1.0, 1.0), (1.0, 0.5, .3)]]
    with self.assertRaises(ValueError):
      ag.MultipleGridAnchorGenerator(box_specs_list)

    # box_specs_list is not a list of lists
    box_specs_list = [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)]
    with self.assertRaises(ValueError):
      ag.MultipleGridAnchorGenerator(box_specs_list)

  def test_invalid_generate_arguments(self):
    base_anchor_size = tf.constant([1.0, 1.0], dtype=tf.float32)
    box_specs_list = [[(1.0, 1.0), (2.0, 1.0), (1.0, 0.5)],
                      [(1.0, 1.0), (1.0, 0.5)]]
    anchor_generator = ag.MultipleGridAnchorGenerator(
        box_specs_list, base_anchor_size)

    # incompatible lengths with box_specs_list
    with self.assertRaises(ValueError):
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                anchor_strides=[(.25, .25)],
                                anchor_offsets=[(.125, .125), (.25, .25)])
    with self.assertRaises(ValueError):
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2), (1, 1)],
                                anchor_strides=[(.25, .25), (.5, .5)],
                                anchor_offsets=[(.125, .125), (.25, .25)])
    with self.assertRaises(ValueError):
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                anchor_strides=[(.5, .5)],
                                anchor_offsets=[(.25, .25)])

    # not pairs
    with self.assertRaises(ValueError):
      anchor_generator.generate(feature_map_shape_list=[(4, 4, 4), (2, 2)],
                                anchor_strides=[(.25, .25), (.5, .5)],
                                anchor_offsets=[(.125, .125), (.25, .25)])
    with self.assertRaises(ValueError):
      anchor_generator.generate(feature_map_shape_list=[(4, 4), (2, 2)],
                                anchor_strides=[(.25, .25, .1), (.5, .5)],
                                anchor_offsets=[(.125, .125),
                                                (.25, .25)])
    with self.assertRaises(ValueError):
      anchor_generator.generate(feature_map_shape_list=[(4), (2, 2)],
                                anchor_strides=[(.25, .25), (.5, .5)],
                                anchor_offsets=[(.125), (.25)])


class FpnAnchorGeneratorTest(tf.test.TestCase):
  def test_construct_pyramid(self):
    scales = [2.0, 1.0, 0.5, 0.25, 0.125]

    anchor_generator = ag.FpnAnchorGenerator(
        scales=scales,
        aspect_ratios=[0.5, 1.0, 2.0],
        base_anchor_size=(256, 256))

    self.assertEqual(len(anchor_generator.num_anchors_per_location()), 5)

    anchor_strides=[(x, x) for x in [64, 32, 16, 8, 4]]
    feature_map_shape_list = [(16, 32), (32, 64), (64, 128),
                              (128, 256), (256, 512)]

    anchor_boxlist = anchor_generator.generate(
        feature_map_shape_list=feature_map_shape_list,
        anchor_strides=anchor_strides)
    anchors = anchor_boxlist.get()
    areas = box_list_ops.area(anchor_boxlist)

    with self.test_session():
      anchors_out = anchors.eval()
      areas_out = areas.eval()
      expected_num_anchors = sum([h * w * 3 for h, w in feature_map_shape_list])
      self.assertEqual(expected_num_anchors, anchors_out.shape[0])
      prev_grid_elems = 0
      for i, grid_size, scale in zip(range(5), feature_map_shape_list, scales):
        this_grid_elems = grid_size[0] * grid_size[1] * 3
        this_anchors_out = anchors_out[prev_grid_elems: this_grid_elems + prev_grid_elems]
        this_areas_out = areas_out[prev_grid_elems: this_grid_elems + prev_grid_elems]
        this_leftmost_corners = [this_anchors_out[1, :], this_anchors_out[4, :]]
        s = anchor_strides[i][0] / 2
        l = (256 * scale) / 2

        expected_areas = np.array([(scale*256)**2] * this_grid_elems)
        expected_leftmost_corners = np.array(
             [[s - l, s - l, s + l, s + l],
              [s - l, 3 * s - l, s + l, 3 * s + l]])

        self.assertAllClose(this_areas_out, expected_areas, rtol=0.1, atol=0.1)
        self.assertAllClose(this_leftmost_corners, expected_leftmost_corners)

        prev_grid_elems += this_grid_elems

  def test_assign_boxes_to_layers(self):
    anchor_generator = ag.FpnAnchorGenerator(
        scales=[2.0, 1.0, 0.5, 0.25, 0.125],
        aspect_ratios=[0.5, 1.0, 2.0],
        base_anchor_size=(256, 256))

    boxes = tf.constant(np.array(
      [(53, 124, 55, 129),
       (145, 40, 195, 60),
       (0, 0, 128, 32),
       (400, 32, 510, 132),
       (256, 0, 550, 212),
       (0, 512, 512, 1024),
       (0, 0, 1024, 2028)]), dtype=tf.float32)
    layer_indices = anchor_generator.assign_boxes_to_layers(boxes)
    expected_layer_indices = [0, 0, 1, 2, 3, 4, 4]

    with self.test_session():
      self.assertAllEqual(layer_indices.eval(), expected_layer_indices)


if __name__ == '__main__':
  tf.test.main()
