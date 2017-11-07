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
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import griddata
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
  invalid = rgb[:, :, 2] == 0
  # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
  out_flow = (rgb[:, :, :2] - 2 ** 15) / 64.0
  print(out_flow.shape, invalid.shape)
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


def _interp_nan(array):
  x = np.arange(0, array.shape[1])
  y = np.arange(0, array.shape[0])
  array = np.ma.masked_invalid(array)
  xx, yy = np.meshgrid(x, y)
  x1 = xx[~array.mask]
  y1 = yy[~array.mask]
  newarr = array[~array.mask]
  return griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear', fill_value=0.0)


def _depth_from_disparity_image(disparity_image, f):
  disparity = disparity_image.astype(np.float32) / 256.
  depth = KITTI_BASELINE_METERS * f / disparity # TODO check units
  depth[disparity_image == 0] = np.nan
  # Interpolate missing values
  depth = _interp_nan(depth)

  plt.imshow(depth[:, :], cmap='gray')
  plt.show()
  return depth


def _create_tfexample(label_map_dict,
                      image_id, encoded_image, encoded_next_image,
                      disparity_image, next_disparity_image, flow):
  #camera_intrinsics = np.array([982.529, 690.0, 233.1966])
  camera_intrinsics = np.array([725.0, 620.5, 187.0], dtype=np.float32)
  f, x0, y0 = camera_intrinsics
  depth = _depth_from_disparity_image(disparity_image, f)
  next_depth = _depth_from_disparity_image(next_disparity_image, f)

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
  example_infos = []
  for i in range(200):
    n = str(i).zfill(6)
    p = os.path.join(dataset_dir, 'training')
    example_infos.append((
        i,
        os.path.join(p, 'image_2', n + '_10.png'),
        os.path.join(p, 'image_2', n + '_11.png'),
        os.path.join(p, 'disp_occ_0', n + '_10.png'),
        os.path.join(p, 'disp_occ_1', n + '_10.png'),
        os.path.join(p, 'flow_occ', n + '_10.png')
    ))

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
        (frame_id,
         image_fn, next_image_fn, depth_fn, next_depth_fn, flow_fn
         ) = example_infos[i]

        if i % 1 == 0:
          sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
              i + 1, len(example_infos), shard_id))
          sys.stdout.flush()

        image_id = str(frame_id)
        image = _read_image(image_fn)
        next_image = _read_image(next_image_fn)
        depth = _read_disparity_image(depth_fn)
        next_depth = _read_disparity_image(next_depth_fn)
        flow = _read_flow(flow_fn)

        example, num_instances = _create_tfexample(
            label_map_dict,
            image_id, image, next_image, depth, next_depth, flow)

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
