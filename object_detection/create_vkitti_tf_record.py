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
import csv

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory for datasets.')
flags.DEFINE_string('set', 'train', 'Create train, val or test set')
flags.DEFINE_string('output_dir', '', 'Root directory for TFRecords')
flags.DEFINE_string('label_map_path', 'data/vkitti_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('examples_per_tfrecord', 100,
                     'How many examples per out file')
FLAGS = flags.FLAGS


def _read_flow(flow_fn):
  "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
  # read png to bgr in 16 bit unsigned short
  bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w, _c = bgr.shape
  assert bgr.dtype == np.uint16 and _c == 3
  # b == invalid flow flag == 0 for sky or other invalid flow
  invalid = bgr[..., 0] == 0
  # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
  out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
  out_flow[..., 0] *= w - 1
  out_flow[..., 1] *= h - 1
  out_flow[invalid] = np.nan # 0 or another value (e.g., np.nan)
  return out_flow


def _read_image(filename, rgb=False):
  "Read (h, w, 3) image from .png."
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w, _c = image.shape
  assert bgr.dtype == np.uint8 and _c == 3
  if rgb:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image


def _read_depth(filename):
  "Read (h, w, 1) float32 depth (in meters) from .png."
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w, _c = image.shape
  assert image.dtype == np.uint16 and _c == 1
  depth = image.astype(np.float32) / 100.0
  return depth


def _get_record_filename(record_dir, shard_id, num_shards):
    output_filename = '{:05d}-of-{:05d}.record'.format(shard_id, num_shards - 1)
    return os.path.join(record_dir, output_filename)


def _create_tfexample(label_map_dict,
                      image_id, image, next_image, depth, flow, segmentation,
                      extrinsics_dict, next_extrinsics_dict,
                      tracking_rows, next_tracking_rows,
                      segmentation_color_map):
  next_rows = {row['tid']: row for row in next_rows}

  boxes = []
  masks = []
  classes = []
  motions = []
  for r in tracking_rows:
    nr = next_tids.get(r['tid'])
    label = r['label']
    tid = r['tid']
    # find out which color this object corresponds to in the segmentation image
    seg_r, seg_g, seg_b = segmentation_color_map['{}:{}'.format(label, tid)]
     # ensure object still tracked in next frame and visible in original frame
    if nr is not None and label != 'DontCare':
      box = np.array([r['t'], r['l'], r['b'], r['r']],
                     dtype=np.float64)
      boxes.append(box)
      class_id = label_map_dict[label.lower()]
      classes.append(class_id)
      mask = segmentation[:, :, 0] == seg_r \
          and segmentation[:, :, 1] == seg_g \
          and segmentation[:, :, 2] == seg_b
      mask = mask.astype(np.uint8)[:, :, 0]
      masks.append(mask)
      #w3d h3d l3d x3d y3d z3d ry rx rz
      py = r['y3d']
      px = r['x3d']
      pz = r['z3d']
      py_t = nr['y3d']
      px_t = nr['x3d']
      pz_t = nr['z3d']
      if rows['moving']:
        ry =
        rx =
        rz =
        ty = py_t - py
        tx = px_t - px
        tz = pz_t - pz
      else:
        ry = 0.0
        rx = 0.0
        rz = 0.0
        ty = 0.0
        tx = 0.0
        tz = 0.0
      motion = np.array([py, px, pz, ry, rx, rz, ty, tx, tz]. dtype=np.float32)
      motions.append(motion)

  if len(boxes) > 0:
      boxes = np.stack(boxes, axis=0)
      masks = np.stack(masks, axis=0)
      motions = np.stack(masks, axis=0)
  else:
      boxes = np.zeros((0, 5))
      masks = np.zeros((0, 0, 0))
      motions = np.zeros((0, 9))

  height, width = image.shape[:2]
  num_instances = boxes.shape[0]

  ymins = (boxes[:, 0] / height).tolist()
  xmins = (boxes[:, 1] / width).tolist()
  ymaxs = (boxes[:, 2] / height).tolist()
  xmaxs = (boxes[:, 3] / width).tolist()
  index_0, index_1, index_2 = np.nonzero(masks)
  encoded_image = cv2.imencode('png', image)
  encoded_next_image = cv2.imencode('png', next_image)
  key = hashlib.sha256(encoded_image).hexdigest()
  extrinsics = np.reshape(
      np.array(extrinsics_dict.values(), dtype=np.float32), [4, 4])

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(image_id.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image),
    'next_image/encoded': dataset_util.bytes_feature(encoded_next_image),
    'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/motion/py': dataset_util.float_list_feature(motions[:, 0].tolist()),
    'image/object/motion/px': dataset_util.float_list_feature(motions[:, 1].tolist()),
    'image/object/motion/pz': dataset_util.float_list_feature(motions[:, 2].tolist()),
    'image/object/motion/ry': dataset_util.float_list_feature(motions[:, 3].tolist()),
    'image/object/motion/rx': dataset_util.float_list_feature(motions[:, 4].tolist()),
    'image/object/motion/rz': dataset_util.float_list_feature(motions[:, 5].tolist()),
    'image/object/motion/ty': dataset_util.float_list_feature(motions[:, 6].tolist()),
    'image/object/motion/tx': dataset_util.float_list_feature(motions[:, 7].tolist()),
    'image/object/motion/tz': dataset_util.float_list_feature(motions[:, 8].tolist()),
    'image/segmentation/object/count': dataset_util.int64_feature(num_instances),
    'image/segmentation/object/index_0': dataset_util.int64_list_feature(index_0.tolist()),
    'image/segmentation/object/index_1': dataset_util.int64_list_feature(index_1.tolist()),
    'image/segmentation/object/index_2': dataset_util.int64_list_feature(index_2.tolist()),
    'image/segmentation/object/class': dataset_util.int64_list_feature(classes),
    'image/depth': dataset_util.bytes_feature(depth.tostring()),
    'image/flow': dataset_util.bytes_feature(flow.tostring()),
    'image/camera/motion': dataset_util.bytes_feature(camera_motion.tostring())
  }))
  return example, num_instances


def _write_tfrecord(record_dir, dataset_dir, split_name, label_map_dict,
                    is_training=False):
  """Loads images and ground truth to a TFRecord.
  Note: masks and bboxes will lose shape info after converting to string.
  """
  vkitti_prefix = 'vkitti_1.3.1_'

  extrinsics_dir = os.path.join(dataset_dir, vkitti_prefix + )
   = os.path.join(dataset_dir, vkitti_prefix + '')
  segmentation_dir = os.path.join(dataset_dir, vkitti_prefix + '')

  styles = ['clone']

  def _collect_image_sequences(suffix):
    type_dir = os.path.join(dataset_dir, vkitti_prefix + suffix)
    seqs = []
    for seq_name in sorted(os.listdir(type_dir)):
      if os.path.isdir(seq_name):
        seq_dir = os.path.join(type_dir, seq_name)
        for style_name in sorted(os.listdir(seq_dir)):
          if style_name in styles:
            style_dir = os.path.join(seq_dir, style_name)
            seqs.append(sorted(os.listdir(style_dir)))
    return seqs

  # a seq consits of .png filenames
  image_seqs = _collect_image_sequences('rgb')
  depth_seqs = _collect_image_sequences('depthgt')
  flow_seqs = _collect_image_sequences('flowgt')
  segmentation_seqs = _collect_image_sequences('scenegt')

  def _collect_line_sequences(suffix, frame_field=0):
    type_dir = os.path.join(dataset_dir, vkitti_prefix + suffix)
    seqs = []
    for seq_name in sorted(os.listdir(type_dir)):
      seq_filename = os.path.join(type_dir, seq_name)
      if os.path.isfile(seq_filename):
        seq_filename_name = seq_filename.split('.txt')[0]
        seq_num, style_name, suffix = seq_filename_name.split('_')
        if style_name in styles:
          with open(seq_filename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ')
            if frame_field is None:
              seqs.append([row for row in reader])
            else:
              seq = []
              current_frame = 0
              frame_rows = []
              for row in reader:
                if row.values()[frame_field] != current_frame:
                  current_frame += 1
                  seq.append(frame_rows)
                  del frame_rows[:]
                frame_rows.append(row)
              seqs.append(seq)
    return seqs

  extrinsics_seqs = _collect_line_sequences('extrinsicsgt') # a seq consists of lists of rows
  tracking_seqs = _collect_line_sequences('motgt')
  segmentation_color_map_seqs = _collect_line_sequences('scenegt', frame_field=None) # a seq is a list of rows

  def _seq_total_len(seqs, last_missing=False):
    return sum([len(seq) for seq in seqs])

  assert _seq_total_len(image_seqs) == _seq_total_len(depth_seqs) \
      == _seq_total_len(flow_seqs, True) == _seq_total_len(extrinsics_seqs) \
      == _seq_total_len(tracking_seqs) == _seq_total_len(segmentation_seqs)
  assert len(label_seqs) == len(image_seqs)

  segmentation_color_maps = []
  for rows in segmentation_color_map_seqs:
    color_map = {}
    for row in rows:
      key = row['Category(:id)']
      val = [int(v) for v in [row['r'], row['g'], row['b']]]
      color_map[key] = val
    segmentation_color_maps.append(color_map)


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
           depth_seq[i], flow_seq[i], segmentation_seq[i],
           extrinsics_seq[i], extrinsics_seq[i + 1],
           tracking_seq[i], tracking_seq[i + 1]))

  if is_training:
    random.seed(0)
    random.shuffle(example_infos)

  num_per_shard = FLAGS.examples_per_tfrecord
  num_shards = int(math.ceil(len(example_infos) / float(num_per_shard)))

  print('creating max. {} examples in {} shards with at most {} examples each'
        .format(len(example_infos), num_shards, num_per_shard))

  created_count = 0

  for shard_id in range(num_shards):
    record_filename = _get_record_filename(record_dir, shard_id, num_shards)
    with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
      start_ndx = shard_id * num_per_shard
      end_ndx = min((shard_id + 1) * num_per_shard, len(example_infos))

      for i in range(start_ndx, end_ndx):
        (seq_id, frame_id,
         image_fn, next_image_fn, depth_fn, flow_fn, segmentation_fn,
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
        flow = _read_flow(flow_fn)

        example, num_instances = _create_tfexample(
            label_map_dict,
            image_id, image, next_image, depth, flow, segmentation,
            extrinsics_rows[0], next_extrinsics_rows[0],
            tracking_rows, next_tracking_rows,
            segmentation_color_maps[seq_id])

        if num_instances > 0 or is_training == False:
          created_count += 1
          tfrecord_writer.write(example.SerializeToString())
          if max_records is not None and max_records == created_count:
              print("Created {} examples ({} skipped)."
                    .format(created_count, len(image_ids) - created_count))
              return
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

  assert set_name in ['train', 'val', 'mini'], set_name
  is_training = set_name in ['train', 'mini']

  # if not tf.gfile.Exists(dataset_root):
  #  tf.gfile.MakeDirs(dataset_root)

  # for url in _DATA_URLS:
  #   download_and_uncompress_zip(url, dataset_dir)
  #   TODO automatically create mini split by copying test/bonn

  record_dir = os.path.join(records_root, 'vkitti_' + set_name)

  if not tf.gfile.Exists(record_dir):
    tf.gfile.MakeDirs(record_dir)

  _write_tfrecord(record_dir,
                  os.path.join(dataset_root, 'vkitti'),
                  split_name,
                  label_map_dict,
                  is_training=is_training)

  print("\nFinished creating Virtual KITTI '{}' set".format(set_name))


if __name__ == '__main__':
  tf.app.run()
