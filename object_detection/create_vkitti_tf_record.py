# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------

import os
import sys
import math
import random
import json
import cv2
import hashlib

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

import cityscapesscripts.helpers.labels as labels

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory for datasets.')
flags.DEFINE_string('set', 'train', 'Create train, val or test set')
flags.DEFINE_string('output_dir', '', 'Root directory for TFRecords')
flags.DEFINE_string('label_map_path', 'data/cityscapes_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('examples_per_tfrecord', 100,
                     'How many examples per out file')
FLAGS = flags.FLAGS


def _read_raw(paths):
    path_queue = tf.train.string_input_producer(
        paths, shuffle=False, capacity=len(paths), num_epochs=1)
    reader = tf.WholeFileReader()
    _, raw = reader.read(path_queue)
    return raw


def _read_image(paths, dtype, channels=1):
    raw = _read_raw(paths)
    return tf.image.decode_png(raw, channels=channels, dtype=dtype)


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


def _collect_files(modality_dir):
    paths = []
    for city_name in sorted(os.listdir(modality_dir)):
        city_dir = os.path.join(modality_dir, city_name)
        for i, filename in enumerate(sorted(os.listdir(city_dir))):
            path = os.path.join(city_dir, filename)
            paths.append(path)
    return paths


def _create_tfexample(label_map_dict,
                      img_id, encoded_img, encoded_next_img, disparity_img, instance_img,
                      camera, vehicle):
    #b = camera['extrinsic']['baseline']
    #f = (camera['intrinsic']['fx'] + camera['intrinsic']['fy']) / 2.
    #disparity = (disparity_img.astype(np.float32) - 1.) / 256.
    #depth = b * f / disparity
    #depth[disparity_img == 0] = 0
    #x0 = camera['intrinsic']['u0']
    #y0 = camera['intrinsic']['v0']
    # TODO variable for seqs of variable offset... use sequence info to get more exact
    #frame_rate = 8.5
    # TODO calc from sequence data for more precise info when skipping frames - e.g. avg.
    # TODO do we have to use the camera extrinsics to get exact translation / yaw?
    #yaw = vehicle['yawRate'] / frame_rate
    #translation = vehicle['speed'] / frame_rate

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

# TODO extract multiple examples at different framerates per annotated file
def _write_tfrecord(record_dir, dataset_dir, split_name,
                    label_map_dict, is_training=False,
                    max_records=None):
    """Loads image files and writes files to a TFRecord.
    Note: masks and bboxes will lose shape info after converting to string.
    """
    print('processing data from {}'.format(split_name))

    image_dir = os.path.join(dataset_dir, 'leftImg8bit', split_name)
    sequence_dir = os.path.join(dataset_dir, 'sequence', split_name)
    gt_dir = os.path.join(dataset_dir, 'gtFine', split_name)
    disparity_dir = os.path.join(dataset_dir, 'disparity', split_name)
    camera_dir = os.path.join(dataset_dir, 'camera', split_name)
    vehicle_dir = os.path.join(dataset_dir, 'vehicle', split_name)
    sequence_dir = os.path.join(dataset_dir, 'sequence', split_name)

    image_paths = []
    next_paths = []
    image_ids = []
    for city_name in sorted(os.listdir(image_dir)):
        city_dir = os.path.join(image_dir, city_name)
        files = sorted(os.listdir(city_dir))
        print("collecting {} examples from city {}".format(len(files), city_name))
        for i, filename in enumerate(files):
            path = os.path.join(city_dir, filename)
            image_paths.append(path)
            # filename should be {city_name}_{seq}_{frame}_leftImg8bit.png
            seq, frame = filename.split('_')[1:3]
            frame_int = int(frame.lstrip('0'))
            next_frame = str(frame_int + 2).zfill(6)
            next_file = "{}_{}_{}_leftImg8bit.png".format(city_name, seq, next_frame)
            next_path = os.path.join(sequence_dir, city_name, next_file)
            assert os.path.isfile(next_path)
            next_paths.append(next_path)
            image_ids.append("{}_{}".format(city_name, i))

    instance_paths = []
    for city_dirname in sorted(os.listdir(gt_dir)):
        city_dir = os.path.join(gt_dir, city_dirname)
        for filename in sorted(os.listdir(city_dir)):
            path = os.path.join(city_dir, filename)
            if path.endswith('instanceIds.png'):
                instance_paths.append(path)

    vehicle_paths = _collect_files(vehicle_dir)
    camera_paths = _collect_files(camera_dir)
    disparity_paths = _collect_files(disparity_dir)

    assert len(image_paths) == len(vehicle_paths) == len(camera_paths) == len(disparity_paths) \
           == len(next_paths) == len(instance_paths)

    if is_training:
        zipped = list(zip(image_paths, image_ids, instance_paths))
        random.seed(0)
        random.shuffle(zipped)
        image_paths, image_ids, instance_paths = zip(*zipped)

    num_per_shard = FLAGS.examples_per_tfrecord
    num_shards = int(math.ceil(len(image_ids) / float(num_per_shard)))

    print('creating max. {} examples in {} shards with at most {} examples each'
          .format(len(image_ids), num_shards, num_per_shard))

    created_count = 0

    for shard_id in range(num_shards):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            record_filename = _get_record_filename(record_dir, shard_id, num_shards)
            with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(image_ids))

                shard_instance_paths = instance_paths[start_ndx:end_ndx]
                shard_image_paths = image_paths[start_ndx:end_ndx]
                shard_next_paths = next_paths[start_ndx:end_ndx]
                shard_disparity_paths = disparity_paths[start_ndx:end_ndx]

                img_ = _read_raw(shard_image_paths)
                next_img_ = _read_raw(shard_next_paths)
                instance_img_ = _read_image(shard_instance_paths, dtype=tf.uint16)
                disparity_img_ = _read_image(shard_disparity_paths, dtype=tf.uint16)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    tf.train.start_queue_runners()

                    for i in range(start_ndx, end_ndx):
                        if i % 1 == 0:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
                                i + 1, len(image_ids), shard_id))
                            sys.stdout.flush()

                        img_id = image_ids[i]
                        img, instance_img, disparity_img, next_img = sess.run(
                            [img_, instance_img_, disparity_img_, next_img_])

                        with open(camera_paths[i]) as camera_file:
                            camera = json.load(camera_file)

                        with open(vehicle_paths[i]) as vehicle_file:
                            vehicle = json.load(vehicle_file)

                        example, num_instances = _create_tfexample(
                            label_map_dict,
                            img_id, img, next_img, disparity_img,
                            instance_img, camera, vehicle)

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
          .format(created_count, len(image_ids) - created_count))
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(_):
    set_name = FLAGS.set
    records_root = FLAGS.output_dir
    dataset_root = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    assert set_name in ['train', 'val', 'test', 'mini'], set_name
    is_training = set_name in ['train', 'mini']

    if set_name == 'mini':
        split_name = 'val'
        max_records = 100
    else:
        split_name = set_name
        max_records = None

    # if not tf.gfile.Exists(dataset_root):
    #  tf.gfile.MakeDirs(dataset_root)

    # for url in _DATA_URLS:
    #   download_and_uncompress_zip(url, dataset_dir)
    #   TODO automatically create mini split by copying test/bonn

    record_dir = os.path.join(records_root, 'cityscapes_' + set_name)

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    _write_tfrecord(record_dir,
                    os.path.join(dataset_root, 'cityscapes'),
                    split_name,
                    label_map_dict,
                    is_training=is_training,
                    max_records=max_records)

    print("\nFinished creating cityscapes '{}' set".format(set_name))


if __name__ == '__main__':
  tf.app.run()
