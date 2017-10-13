
Download and extract http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz to data

``` bash
protoc object_detection/protos/*.proto --python_out=.
python create_cityscapes_tf_record.py --data_dir=MY_DATASET_ROOT --output_dir=data/records --set train
python create_cityscapes_tf_record.py --data_dir=MY_DATASET_ROOT --output_dir=data/records --set val
```

python train.py \
  --logtostderr \
  --pipeline_config_path=data/configs/faster_rcnn_resnet50_cityscapes.config \
  --train_dir=output/train/test

python eval.py \
    --logtostderr \
    --pipeline_config_path=data/configs/faster_rcnn_resnet50_cityscapes.config \
    --checkpoint_dir=output/train/test \
    --eval_dir=output/eval/test




python train.py --logtostderr --pipeline_config_path=data/configs/mask_rcnn_resnet50_fpn_cityscapes.config --train_dir=output/train/mask_rcnn_fpn --gpu 0

python eval.py --logtostderr --pipeline_config_path=data/configs/mask_rcnn_resnet50_fpn_cityscapes.config --checkpoint_dir=output/train/mask_rcnn_fpn --eval_dir=output/eval/mask_rcnn_fpn


modified tests:

python -m object_detection.core.box_predictor_test
python -m object_detection.models.faster_rcnn_resnet_v1_fpn_feature_extractor_test
python -m object_detection.anchor_generators.multiple_grid_anchor_generator_test
python -m object_detection.meta_architectures.faster_rcnn_meta_arch_test
