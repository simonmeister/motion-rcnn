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
from object_detection.utils.np_motion_util import dense_flow_from_motion, euler_to_rot, _rotation_angle


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory for datasets.')
flags.DEFINE_string('set', 'train', 'Create train or val set')
flags.DEFINE_string('output_dir', '', 'Root directory for TFRecords')
flags.DEFINE_string('label_map_path', 'data/vkitti_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('examples_per_tfrecord', 100,
                     'How many examples per out file')
flags.DEFINE_boolean('gt_rigid_flow_from_motion', True,
                     'Use supplied flow gt or compose from motion '
                     '(more precise, but some motions like wheels are not captured)')
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


def _read_depth(filename):
  "Read (h, w, 1) float32 depth (in meters) from .png."
  image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  h, w = image.shape[:2]
  assert image.dtype == np.uint16 and len(image.shape) == 2
  depth = image.astype(np.float32) / 100.0
  return depth


def _get_record_filename(record_dir, shard_id, num_shards):
    output_filename = '{:05d}-of-{:05d}.record'.format(shard_id, num_shards - 1)
    return os.path.join(record_dir, output_filename)


def _euler_to_rot(dct):
  x = dct['rx']
  y = dct['ry']
  z = dct['rz']
  return euler_to_rot(x, y, z)


def _get_pivot(dct):
  return np.array([dct['x3d'], dct['y3d'], dct['z3d']], dtype=np.float32)


def _Rt_to_hom(R, t):
  t = np.expand_dims(t, axis=1)
  h = np.concatenate([R, t], axis=1)
  last_row = np.expand_dims(np.array([0.0, 0.0, 0.0, 1.0]), axis=0)
  return np.concatenate([h, last_row], axis=0)


def _hom_to_Rt(hom):
  return

def _p_to_hom(p):
  return np.concatenate([p, np.array([1.0])])


def _hom_to_p(p_hom):
  return p_hom[:3] / p_hom[3]


def _create_tfexample(label_map_dict,
                      image_id, encoded_image, encoded_next_image,
                      depth, next_depth, flow, segmentation,
                      extrinsics_dict, next_extrinsics_dict,
                      tracking_rows, next_tracking_rows,
                      segmentation_color_map, first_extrinsics_dict):
  frame_id = int(image_id.split('_')[1])
  assert frame_id == extrinsics_dict['frame'] == next_extrinsics_dict['frame'] - 1
  next_tracking_row_map = {row['tid']: row for row in next_tracking_rows}
  height, width = depth.shape[:2]

  extrinsics = np.reshape(
      np.array(list(extrinsics_dict.values())[1:], dtype=np.float32), [4, 4])
  next_extrinsics = np.reshape(
      np.array(list(next_extrinsics_dict.values())[1:], dtype=np.float32), [4, 4])
  first_extrinsics = np.reshape(
      np.array(list(first_extrinsics_dict.values())[1:], dtype=np.float32), [4, 4])
  camera_moving = not np.allclose(extrinsics, next_extrinsics)
  rot_cam1 = extrinsics[:3, :3]
  rot_cam2 = next_extrinsics[:3, :3]
  trans_cam1 = extrinsics[:3, 3]
  trans_cam2 = next_extrinsics[:3, 3]
  rot_cam2_to_cam1 = rot_cam1 @ rot_cam2.T
  rot_cam1_to_cam2 = rot_cam2_to_cam1.T
  trans_cam1_to_cam2 = trans_cam2 - rot_cam1_to_cam2 @ trans_cam1
  camera_motion = np.concatenate([rot_cam1_to_cam2.ravel(),
                                  trans_cam1_to_cam2.ravel(),
                                  np.array([camera_moving], dtype=np.float32)])
  cam2_to_cam1_hom = extrinsics @ np.linalg.inv(next_extrinsics)


  boxes = []
  masks = []
  classes = []
  motions = []
  diff = 0
  for row in tracking_rows:
    next_row = next_tracking_row_map.get(row['tid'])
    label = row['orig_label']
    tid = row['tid']
     # ensure object still tracked in next frame and visible in original frame
    if next_row is not None and row['occupr'] > 0.1:
      assert frame_id == row['frame'] == next_row['frame'] - 1
      box = np.array([row['t'], row['l'], row['b'], row['r']],
                     dtype=np.float64)
      boxes.append(box)
      class_id = label_map_dict[label.lower()]
      classes.append(class_id)
      # find out which color this object corresponds to in the segmentation image
      seg_r, seg_g, seg_b = segmentation_color_map['{}:{}'.format(label, tid)]
      mask = ((segmentation[:, :, 0] == seg_r).astype(np.uint8) +
          (segmentation[:, :, 1] == seg_g).astype(np.uint8) +
          (segmentation[:, :, 2] == seg_b).astype(np.uint8))
      mask = (mask == 3).astype(np.uint8)
      masks.append(mask)
      moving = int(row['moving'])
      p1 = _get_pivot(row)
      p2 = _get_pivot(next_row)
      r1 = _euler_to_rot(row)
      r2 = _euler_to_rot(next_row)
      hom1 = _Rt_to_hom(r1, p1)
      hom2 = _Rt_to_hom(r2, p2)
      #obj_1_to_2 = hom2 @ np.linalg.inv(hom1)
      #obj_1_to_2 = hom2 @ np.linalg.inv(extrinsics) @ next_extrinsics @ np.linalg.inv(hom1)
      #print(obj_1_to_2)
      #r1_to_r2, p1_to_p2 = _hom_to_Rt(obj_1_to_2)
      r2 = rot_cam2_to_cam1 @ r2
      r1_to_r2 = r2 @ r1.T
      #print(r1_to_r2)
      #r1_to_r2 = r2_cam2 @ rot_cam2 @ rot_cam1.T @ r1.T
      p2_hom = np.concatenate([p2, np.array([1])])
      p2_cam1_hom = cam2_to_cam1_hom @ p2_hom
      p2_cam1 = p2_cam1_hom[:3] / p2_cam1_hom[3]
      p1_to_p2 = p2_cam1 - (r1_to_r2 @ p1)
      if moving == 0:
        #if not np.allclose(p1_to_p2, np.zeros_like(p1_to_p2), atol=1e-4):
        #  print('trans', np.mean(p1_to_p2))
        #if not np.allclose(r1_to_r2, np.eye(3), atol=1e-2):
        #  print('rot', np.arccos(np.clip((np.trace(r1_to_r2, axis1=0, axis2=1) - 1) / 2, -1, 1)))
        r1_to_r2 = np.eye(3, dtype=np.float32)
        p1_to_p2 = np.zeros([3], dtype=np.float32)
      motion = np.concatenate([r1_to_r2.ravel(), p1_to_p2, p1,
                               np.array([moving], dtype=np.float32)])
      if moving == 1:
        diff += np.sum(np.abs(rot_cam1_to_cam2 @ ((r1_to_r2 @ p1) + p1_to_p2) + trans_cam1_to_cam2 - p2))
        #diff += np.sum(np.abs(_hom_to_p(obj_1_to_2 @ _p_to_hom(p1)) - p2))
      motions.append(motion)
  print(diff)
  if len(boxes) > 0:
      boxes = np.stack(boxes, axis=0)
      masks = np.stack(masks, axis=0)
      motions = np.stack(motions, axis=0)
  else:
    boxes = np.zeros((0, 5), dtype=np.float32)
    masks = np.zeros((0, height, width), dtype=np.float32)
    motions = np.zeros((0, 15), dtype=np.float32)

  num_instances = boxes.shape[0]

  ymins = (boxes[:, 0] / height).tolist()
  xmins = (boxes[:, 1] / width).tolist()
  ymaxs = (boxes[:, 2] / height).tolist()
  xmaxs = (boxes[:, 3] / width).tolist()
  index_0, index_1, index_2 = np.nonzero(masks)
  key = hashlib.sha256(encoded_image).hexdigest()

  camera_intrinsics = np.array([725.0, 620.5, 187.0], dtype=np.float32)

  if FLAGS.gt_rigid_flow_from_motion:
    example_flow = dense_flow_from_motion(np.expand_dims(depth, 2), motions, masks,
                                          camera_motion, camera_intrinsics)
  else:
    example_flow = flow

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
    'image/object/motion': dataset_util.float_list_feature(motions.ravel().tolist()),
    'image/segmentation/object/count': dataset_util.int64_feature(num_instances),
    'image/segmentation/object/index_0': dataset_util.int64_list_feature(index_0.tolist()),
    'image/segmentation/object/index_1': dataset_util.int64_list_feature(index_1.tolist()),
    'image/segmentation/object/index_2': dataset_util.int64_list_feature(index_2.tolist()),
    'image/segmentation/object/class': dataset_util.int64_list_feature(classes),
    'image/depth': dataset_util.float_list_feature(depth.ravel().tolist()),
    'next_image/depth': dataset_util.float_list_feature(next_depth.ravel().tolist()),
    'image/flow': dataset_util.float_list_feature(example_flow.ravel().tolist()),
    'image/camera/motion': dataset_util.float_list_feature(camera_motion.tolist()),
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

  def _collect_line_sequences(suffix, frame_field=0):
    """If frame_field=None, returns a list containing a list of rows for each file.
    Otherwise, returns (a list containing) one list per sequence/file,
    each containing one list of rows for each frame in the sequence.
    A row is stored as orderect dict."""
    type_dir = os.path.join(dataset_dir, vkitti_prefix + suffix)
    seqs = []
    for seq_name in sorted(os.listdir(type_dir)):
      seq_filename = os.path.join(type_dir, seq_name)
      if os.path.isfile(seq_filename):
        seq_name_parts = seq_name.split('.txt')[0].split('_')
        seq_num, style_name = seq_name_parts[:2]
        if style_name in styles:
          with open(seq_filename) as csvfile:
            data_frame = pd.read_csv(csvfile, sep=' ', index_col=False)
            rows = []
            for nt in data_frame.itertuples():
              od = nt._asdict()
              del od['Index']
              rows.append(od)
            if frame_field is None:
              seqs.append(rows)
            else:
              seq = []
              current_frame = 0
              frame_rows = []
              for row in rows:
                row_frame = list(row.values())[frame_field]
                if row_frame > current_frame:
                  # padding for frames without objects
                  frame_step = row_frame - current_frame
                  if frame_step > 1:
                    padding = frame_step - 1
                    for _ in range(padding):
                      seq.append([])
                  seq.append(frame_rows)
                  frame_rows = []
                  current_frame = row_frame
                frame_rows.append(row)
              if len(frame_rows) > 0:
                seq.append(frame_rows)
              seqs.append(seq)
    return seqs

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

  segmentation_color_maps = []
  for rows in segmentation_color_map_seqs:
    color_map = {}
    for row in rows:
      key = row['_1']
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
           depth_seq[i], depth_seq[i + 1], flow_seq[i], segmentation_seq[i],
           extrinsics_seq[i], extrinsics_seq[i + 1],
           tracking_seq[i], tracking_seq[i + 1],
           extrinsics_seq[0]))

  random.seed(0)
  random.shuffle(example_infos)
  if split_name == 'val':
    example_infos = example_infos[:100]
  else:
    example_infos = example_infos[100:]

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
         tracking_rows, next_tracking_rows, first_extrinsics_rows) = example_infos[i]

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
            segmentation_color_maps[seq_id],
            first_extrinsics_rows[0])

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

  assert set_name in ['train', 'val', 'mini'], set_name
  is_training = set_name in ['train', 'mini']
  split_name = 'train' if set_name == 'train' else 'val'

  # if not tf.gfile.Exists(dataset_root):
  #  tf.gfile.MakeDirs(dataset_root)

  # for url in _DATA_URLS:
  #   download_and_uncompress_zip(url, dataset_dir)

  record_dir = os.path.join(records_root, 'vkitti_' + set_name)
  if os.path.isdir(record_dir):
    shutil.rmtree(record_dir)

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
