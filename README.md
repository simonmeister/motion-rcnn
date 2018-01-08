# Motion R-CNN

This repository contains the TensorFlow prototype implementation of my bachelor thesis
[Motion R-CNN: Instance-level 3D Motion Estimation with Region-based CNNs](
https://drive.google.com/open?id=18hSyz2Wgd-cb-Psju5_oPUyZyE3T3k7j).

In addition to the functionality provided by the TensorFlow Object Detection API (at the time of writing), the code supports:
- prediction of instance masks
- prediction of 3D camera motion
- prediction of 3D instance motions
- Feature Pyramid Networks

Note that the code only supports training on the Virtual KITTI dataset,
but it is easy to adapt it to other datasets.
Motion prediction is fully optional and the code can be used as a Mask R-CNN
implementation.
Support for cityscapes is implemented, but using the records created with `create_citiscapes_tf_record.py` 
may required adapting the `data_decoder` as the record interface changed.

### License

Motion R-CNN is released under the MIT License (refer to the LICENSE file for details).

## Usage

### Requirements
- [tensorflow (>= 1.3.0)](https://www.tensorflow.org/install/install_linux) with GPU support.
- sudo apt-get install protobuf-compiler
- `pip install opencv-python pandas pillow lxml matplotlib`

### Setup
- from the project root directory, run ``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``
- download and extract the 
[pre-trained ResNet-50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz )
model to `object_detection/data`
- download all of the 
[Virtual KITTI](http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds)
ground truth and extract the folders into a directory named `vkitti`.
- cd to the project root directory
- run `protoc object_detection/protos/*.proto --python_out=.` 
- run `python create_vkitti_tf_record.py --data_dir=<data_parent_dir> --output_dir=data/records --set val`
- run `python create_vkitti_tf_record.py --data_dir=<data_parent_dir> --output_dir=data/records --set train`

Note that `<data_parent_dir>` is the directory containing the `vkitti` directory.

### Training & evaluating
Use
- `python train.py --logtostderr --pipeline_config_path=data/configs/motion_rcnn_vkitti_cam.config --train_dir=output/train/motion_rcnn_vkitti_cam --gpu 0`
- `python eval.py --logtostderr --pipeline_config_path=data/configs/motion_rcnn_vkitti_cam.config --checkpoint_dir=output/train/motion_rcnn_vkitti_cam --eval_dir=output/eval/motion_rcnn_vkitti_cam`

to train and evaluate a model with camera and instance motion prediction.
You can adapt the configurations found in `data/configs/`. For a description of the configuration parameters, see `object_detection/protos`.

## Navigating the code

The following files were added or modified from the original Object Detection API code
- [motion_util.py](object_detection/utils/motion_util.py): losses for training the motion estimation and post-processing utilities
- [np_motion_util.py](object_detection/utils/np_motion_util.py): composition of optical flow using motion predictions and performance evaluation utilities
- [create_vkitti_tf_record.py](object_detection/create_vkitti_tf_record.py): process virtual kitti dataset
- [faster_rcnn_meta_arch.py](object_detection/meta_architectures/faster_rcnn_meta_arch.py): adapted to support instance mask, instance motion, and camera motion training.
- [target_assigner.py](object_detection/core/target_assigner.py): updated to support mask and motion target assignment
- [box_predictor.py](object_detection/core/box_predictor.py): updated to support FPN as well as mask and motion prediction
- [post_processing.py](object_detection/core/post_processing.py): updated to pass through instance motions

Additionally, some proto params and builders were modified, and extensions were made to
[eval_util.py](object_detection/eval_util.py),
[eval.py](object_detection/eval.py),
[evaluator.py](object_detection/evaluator.py),
[train.py](object_detection/train.py),
[trainer.py](object_detection/trainer.py).

The following tests were added or modified:
- object_detection.core.box_predictor_test
- object_detection.core.post_processing_test
- object_detection.models.faster_rcnn_resnet_v1_feature_extractor_test
- object_detection.models.faster_rcnn_resnet_v1_fpn_feature_extractor_test
- object_detection.anchor_generators.multiple_grid_anchor_generator_test
- object_detection.meta_architectures.faster_rcnn_meta_arch_test

## Acknowledgments
This repository is based on the
[TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
