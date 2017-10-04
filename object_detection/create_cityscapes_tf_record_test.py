# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import sys
import os
import glob
import shutil

import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt

from cityscapesscripts.helpers.labels import trainId2label

from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder


with tf.Graph().as_default():
    file_pattern = 'object_detection/data/records/cityscapes_mini/00000-of-00004.record'
    tfrecords = glob.glob(file_pattern)

    with tf.device('/cpu:0'):
        filename_queue = tf.train.string_input_producer(
            tfrecords, capacity=len(tfrecords))
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        example = TfExampleDecoder().decode(serialized_example)
        print(example.keys())

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    sess.run(init_op)

    tf.train.start_queue_runners(sess=sess)
    out_dir = 'object_detection/output/tests/cityscapes/'
    if os.path.isdir(out_dir):
      shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    with sess.as_default():
        for i in range(30):
            example_np = sess.run(example)
            img_id_np = example_np['filename']
            image_np = example_np['image']
            gt_boxes_np = example_np['groundtruth_boxes']
            gt_classes_np = example_np['groundtruth_classes']
            gt_masks_np = example_np['groundtruth_instance_masks']
            height, width = image_np.shape[:2]
            #depth_np = example_np['depth']
            #print(np.count_nonzero(np.isinf(depth_np)))
            #import matplotlib.pyplot as plt
            #depth_mask = tf.to_float(
            #    tf.logical_and(
            #        tf.logical_not(tf.is_inf(depth_np)),
            #        depth_np != 0))
            #print(sess.run(depth_mask))
            #print(sess.run(tf.reduce_sum(depth_np * depth_mask)))
            #plt.imshow(depth_np[:, :, 0], cmap='gray')
            #plt.show()
            num_instances_np = gt_masks_np.shape[0]
            print('image_id: {}, instances: {}, shape: {}'
                  .format(img_id_np, num_instances_np, image_np.shape))
            image_np = np.zeros_like(image_np)
            #image_np = np.squeeze(image_np)

            # overlay masks
            for i in range(gt_boxes_np.shape[0]):
                label = trainId2label[gt_classes_np[i]]
                mask = np.expand_dims(gt_masks_np[i, :, :], 2)
                image_np += (0.5 * mask * np.array(label.color)).astype(np.uint8)
            # draw boxes
            im = Image.fromarray(image_np)
            imd = ImageDraw.Draw(im)
            for i in range(gt_boxes_np.shape[0]):
                label = trainId2label[gt_classes_np[i]]
                color = 'rgb({},{},{})'.format(*label.color)
                pos = gt_boxes_np[i, :]
                y0 = pos[0] * height
                x0 = pos[1] * width
                y1 = pos[2] * height
                x1 = pos[3] * width
                imd.rectangle([x0, y0, x1, y1], outline=color)
                imd.text(((x0 + x1) / 2, y1), label.name, fill=color)

            im.save(os.path.join(out_dir, str(img_id_np) + '.png'))
        sess.close()
