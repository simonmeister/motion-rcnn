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


def _read_image(filename):
  "Read (h, w, 3) RGB image from .png."
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w, _c = image.shape
  assert bgr.dtype == np.uint8 and _c == 3
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image


def _read_depth(filename):
  "Read (h, w, 1) float32 depth (in meters) from .png."
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w, _c = image.shape
  assert image.dtype == np.uint16 and _c == 1
  depth = image.astype(np.float32) / 100.0
  return depth


def _get_instance_masks_and_boxes_np(instance_img):
    """Get instance level ground truth.

    Note: instance_img is expected to consist of regular ids, not trainIds.

    Returns:
      masks: (m, h, w) numpy array
      boxes: (m, 5), [[x1, y1, x2, y2], ...]
      classes: (m,) list of class names
    """
    all_ids = np.unique(instance_img).tolist()
    class_ids = [label.id for label in labels.labels]
    pixel_ids_of_instances = [i for i in all_ids if i not in class_ids]

    masks = []
    cropped_masks = []
    boxes = []
    classes = []
    for pixel_id in pixel_ids_of_instances:
        class_id = pixel_id // 1000
        class_name = labels.id2label[class_id].name

        mask = instance_img == pixel_id
        nonzero_y, nonzero_x = np.nonzero(np.squeeze(mask))
        y1 = np.min(nonzero_y)
        y2 = np.max(nonzero_y)
        x1 = np.min(nonzero_x)
        x2 = np.max(nonzero_x)

        box = np.array([x1, y1, x2, y2], dtype=np.float64)
        mask = mask.astype(np.uint8)[:, :, 0]

        masks.append(mask)
        boxes.append(box)
        classes.append(class_name)

    if len(boxes) > 0:
        boxes = np.stack(boxes, axis=0)
        masks = np.stack(masks, axis=0)
    else:
        boxes = np.zeros((0, 5))
        masks = np.zeros((0, 0, 0))
    return masks, boxes, classes


def _get_record_filename(record_dir, shard_id, num_shards):
    output_filename = '{:05d}-of-{:05d}.record'.format(shard_id, num_shards - 1)
    return os.path.join(record_dir, output_filename)


def _create_tfexample(label_map_dict,
                      img_id, encoded_img, encoded_next_img, disparity_img, instance_img,
                      camera, vehicle):

    height, width = instance_img.shape[:2]
    masks, boxes, classes_text = _get_instance_masks_and_boxes_np(instance_img)
    num_instances = boxes.shape[0]

    xmins = (boxes[:, 0] / width).tolist()
    ymins = (boxes[:, 1] / height).tolist()
    xmaxs = (boxes[:, 2] / width).tolist()
    ymaxs = (boxes[:, 3] / height).tolist()
    classes = [label_map_dict[text] for text in classes_text]
    index_0, index_1, index_2 = np.nonzero(masks)
    key = hashlib.sha256(encoded_img).hexdigest()

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(img_id.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(img_id.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_img),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/segmentation/object/count': dataset_util.int64_feature(num_instances),
      'image/segmentation/object/index_0': dataset_util.int64_list_feature(index_0.tolist()),
      'image/segmentation/object/index_1': dataset_util.int64_list_feature(index_1.tolist()),
      'image/segmentation/object/index_2': dataset_util.int64_list_feature(index_2.tolist()),
      'image/segmentation/object/class': dataset_util.int64_list_feature(classes),
      'image/depth': dataset_util.bytes_feature(depth.tostring())
    }))
    return example, num_instances


# 'image/id': _bytes_feature(img_id.encode('utf8')),
# 'image/encoded': _bytes_feature(img.tostring()),
# 'image/height': _int64_feature(height),
# 'image/width': _int64_feature(width),
# 'next_image/encoded': _bytes_feature(next_img.tostring()),
# 'label/num_instances': _int64_feature(num_instances),
# 'label/boxes': _bytes_feature(boxes.tostring()),
# 'label/masks': _bytes_feature(masks.tostring()),
# 'label/depth': _bytes_feature(depth.tostring()),
# 'label/camera/intrinsics/f': _float_feature(f),
# 'label/camera/intrinsics/x0': _float_feature(x0),
# 'label/camera/intrinsics/y0': _float_feature(y0),
# 'label/camera/motion/yaw': _float_feature(yaw),
# 'label/camera/motion/translation': _float_feature(translation),


def _write_tfrecord(record_dir, dataset_dir, split_name, is_training=False):
  """Loads image files and writes files to a TFRecord.
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
            reader = csv.DictReader(csvfile)
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
  label_seqs = _collect_line_sequences('scenegt') # TODO this is not per-image!!

  def _seq_total_len(seqs, last_missing=False):
    return sum([len(seq) for seq in seqs])

  assert _seq_total_len(image_seqs) == _seq_total_len(depth_seqs) \
      == _seq_total_len(flow_seqs, True) == _seq_total_len(extrinsics_seqs) \
      == _seq_total_len(tracking_seqs) == _seq_total_len(segmentation_seqs) \
      == _seq_total_len(label_seqs)

  seq_lists = zip(image_seqs, depth_seqs, flow_seqs, segmentation_seqs,
                  extrinsics_seqs, tracking_seqs, label_seqs)
  example_infos = []
  for (image_seq, depth_seq, flow_seq, segmentation_seq,
       extrinsics_seq, tracking_seq, label_seq) in seq_lists:
    for i in range(len(image_seq) - 1):
      example_infos.append(
          (image_seq[i], image_seq[i + 1], depth_seq[i], flow_seq[i],
           extrinsics_seq[i], tracking_seq[i], label_seq[i]))

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
        (image_fn, next_image_fn, depth_fn, flow_fn, extrinsics_rows,
         tracking_rows, label_TODO) = example_infos[i]

        if i % 1 == 0:
          sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
              i + 1, len(example_infos), shard_id))
          sys.stdout.flush()

        image = _read_image(image_fn)
        next_image = _read_image(next_image_fn)
        depth = _read_image(depth_fn, False)
        flow = _read_flow(flow_fn)
        extrinsics_dict = extrinsics_rows[0]


        example, num_instances = _create_tfexample(
            label_map_dict,
            image_id, image, next_image, depth, flow,
            extrinsics_dict)

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
                  is_training=is_training)

  print("\nFinished creating Virtual KITTI '{}' set".format(set_name))


if __name__ == '__main__':
  tf.app.run()
