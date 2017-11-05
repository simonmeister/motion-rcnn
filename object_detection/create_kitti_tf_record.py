# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------

import os
import sys
import math
import random
import cv2
import hashlib
import pandas as pd
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_motion_util import dense_flow_from_motion, euler_to_rot


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory for datasets.')
flags.DEFINE_string('set', 'train', 'Create train or val set')
flags.DEFINE_string('output_dir', '', 'Root directory for TFRecords')
flags.DEFINE_string('label_map_path', 'data/vkitti_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('examples_per_tfrecord', 100,
                     'How many examples per out file')
FLAGS = flags.FLAGS


KITTI_BASELINE_METERS = 0.54


def _read_flow(flow_fn):
  "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
  # read png to bgr in 16 bit unsigned short
  bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  h, w, _c = rgb.shape
  assert rgb.dtype == np.uint16 and _c == 3
  invalid = rgb[:, 2] == 0
  # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
  out_flow = (rgb[:, :2] - 2 ** 15) / 64.0
  out_flow[invalid] = np.nan # 0 or another value (e.g., np.nan)
  return out_flow


def _read_image(filename, rgb=False):
  "Read (h, w, 3) image from .png."
  if not rgb:
    with open(filename, 'rb') as f:
      image = f.read()
    return image
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w, _c = image.shape
  assert image.dtype == np.uint8 and _c == 3
  if rgb:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image


def _read_disparity_image(filename):
  "Read (h, w, 1) uint16 KITTI disparity image from .png."
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w = image.shape[:2]
  assert image.dtype == np.uint16 and len(image.shape) == 2
  return image


def _get_record_filename(record_dir, shard_id, num_shards):
    output_filename = '{:05d}-of-{:05d}.record'.format(shard_id, num_shards - 1)
    return os.path.join(record_dir, output_filename)


def _depth_from_disparity_image(disparity_image, f, b):
  disparity = disparity_img.astype(np.float32) / 256.
  depth = KITTI_BASELINE_METERS * f / disparity # TODO check units
  depth[disparity_img == 0] = np.nan


def _create_tfexample(label_map_dict,
                      image_id, encoded_image, encoded_next_image,
                      disparity_image, next_disparity_image, flow,
                      camera_intrinsics):
  f, x0, y0 = camera_intrinsics
  depth = _depth_from_disparity(disparity, f, b)
  next_depth = _depth_from_disparity(next_disparity, f, b)

  key = hashlib.sha256(encoded_image).hexdigest()
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(image_id.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image),
    'next_image/encoded': dataset_util.bytes_feature(encoded_next_image),
    'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/depth': dataset_util.float_list_feature(depth.ravel().tolist()),
    'next_image/depth': dataset_util.float_list_feature(next_depth.ravel().tolist()),
    'image/flow': dataset_util.float_list_feature(example_flow.ravel().tolist()),
    'image/camera/intrinsics': dataset_util.float_list_feature(camera_intrinsics.tolist())
  }))
  return example, num_instances


def _write_tfrecord(record_dir, dataset_dir, split_name, label_map_dict,
                    is_training=False):
  """Loads images and ground truth to a TFRecord.
  Note: masks and bboxes will lose shape info after converting to string.
  """
  vkitti_prefix = 'vkitti_1.3.1_'

  styles = ['clone']

  def _collect_image_sequences(suffix):
    type_dir = os.path.join(dataset_dir, vkitti_prefix + suffix)
    seqs = []
    for seq_name in sorted(os.listdir(type_dir)):
      seq_dir = os.path.join(type_dir, seq_name)
      if os.path.isdir(seq_dir):
        for style_name in sorted(os.listdir(seq_dir)):
          if style_name in styles:
            style_dir = os.path.join(seq_dir, style_name)
            seqs.append([os.path.join(style_dir, image_name)
                         for image_name in sorted(os.listdir(style_dir))])
    return seqs

  # a seq consits of .png filenames
  image_seqs = _collect_image_sequences('rgb')
  depth_seqs = _collect_image_sequences('depthgt')
  flow_seqs = _collect_image_sequences('flowgt')
  segmentation_seqs = _collect_image_sequences('scenegt')

  def _pad_trailing(seqs, ref_seqs):
    for seq, ref_seq in zip(seqs, ref_seqs):
      padding = len(ref_seq) - len(seq)
      for _ in range(padding):
        seq.append([])

  extrinsics_seqs = _collect_line_sequences('extrinsicsgt')
  tracking_seqs = _collect_line_sequences('motgt')
  segmentation_color_map_seqs = _collect_line_sequences('scenegt', frame_field=None)
  _pad_trailing(tracking_seqs, extrinsics_seqs)

  def _seq_total_len(seqs, last_missing=False):
    return sum([len(seq) + (1 if last_missing else 0) for seq in seqs])

  assert _seq_total_len(image_seqs) == _seq_total_len(depth_seqs) \
      == _seq_total_len(flow_seqs, True) == _seq_total_len(extrinsics_seqs) \
      == _seq_total_len(tracking_seqs) == _seq_total_len(segmentation_seqs)
  assert len(segmentation_color_map_seqs) == len(image_seqs)

  seq_lists = zip(image_seqs, depth_seqs, flow_seqs, segmentation_seqs,
                  extrinsics_seqs, tracking_seqs)

  example_infos = []
  for seq_i, seq_list in enumerate(seq_lists):
    (image_seq, depth_seq, flow_seq, segmentation_seq,
     extrinsics_seq, tracking_seq) = seq_list
    for i in range(len(image_seq) - 1):
      example_infos.append(
          (seq_i, i,
           image_seq[i], image_seq[i + 1],
           depth_seq[i], depth_seq[i + 1], flow_seq[i], segmentation_seq[i],
           extrinsics_seq[i], extrinsics_seq[i + 1],
           tracking_seq[i], tracking_seq[i + 1]))

  if is_training:
    random.seed(0)
    random.shuffle(example_infos)
  #if split_name == 'val':
  #  example_infos = example_infos[:100]
  #else:
  #  example_infos = example_infos[100:]

  num_per_shard = FLAGS.examples_per_tfrecord
  num_shards = int(math.ceil(len(example_infos) / float(num_per_shard)))

  print('Creating {} examples in {} shards with at most {} examples each'
        .format(len(example_infos), num_shards, num_per_shard))

  created_count = 0

  for shard_id in range(num_shards):
    record_filename = _get_record_filename(record_dir, shard_id, num_shards)
    with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
      start_ndx = shard_id * num_per_shard
      end_ndx = min((shard_id + 1) * num_per_shard, len(example_infos))

      for i in range(start_ndx, end_ndx):
        (seq_id, frame_id,
         image_fn, next_image_fn, depth_fn, next_depth_fn, flow_fn, segmentation_fn,
         extrinsics_rows, next_extrinsics_rows,
         tracking_rows, next_tracking_rows) = example_infos[i]

        if i % 1 == 0:
          sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
              i + 1, len(example_infos), shard_id))
          sys.stdout.flush()

        image_id = '{}_{}'.format(seq_id, frame_id)
        image = _read_image(image_fn)
        next_image = _read_image(next_image_fn)
        segmentation = _read_image(segmentation_fn, rgb=True)
        depth = _read_depth(depth_fn)
        next_depth = _read_depth(next_depth_fn)
        flow = _read_flow(flow_fn)

        example, num_instances = _create_tfexample(
            label_map_dict,
            image_id, image, next_image, depth, next_depth, flow, segmentation,
            extrinsics_rows[0], next_extrinsics_rows[0],
            tracking_rows, next_tracking_rows,
            segmentation_color_maps[seq_id])

        if num_instances > 0 or is_training == False:
          created_count += 1
          tfrecord_writer.write(example.SerializeToString())
        else:
          print("Skipping example {}: no instances".format(i))

  print("Created {} examples ({} skipped)."
        .format(created_count, len(example_infos) - created_count))
  sys.stdout.write('\n')
  sys.stdout.flush()


def main(_):
  set_name = FLAGS.set
  records_root = FLAGS.output_dir
  dataset_root = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  assert set_name in ['train'], set_name
  is_training = set_name in ['train']
  split_name = 'train' # if set_name == 'train' else 'val'

  # if not tf.gfile.Exists(dataset_root):
  #  tf.gfile.MakeDirs(dataset_root)

  # for url in _DATA_URLS:
  #   download_and_uncompress_zip(url, dataset_dir)

  record_dir = os.path.join(records_root, 'kitti_' + set_name)
  if os.path.isdir(record_dir):
    shutil.rmtree(record_dir)

  if not tf.gfile.Exists(record_dir):
    tf.gfile.MakeDirs(record_dir)

  _write_tfrecord(record_dir,
                  os.path.join(dataset_root, 'data_scene_flow'),
                  split_name,
                  label_map_dict,
                  is_training=is_training)

  print("\nFinished creating KITTI '{}' set".format(set_name))


if __name__ == '__main__':
  tf.app.run()
