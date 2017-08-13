
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




python train.py --logtostderr --pipeline_config_path=data/configs/mask_rcnn_resnet50_cityscapes.config --train_dir=output/train/mask_rcnn --gpu 0

python eval.py --logtostderr --pipeline_config_path=data/configs/mask_rcnn_resnet50_cityscapes.config --checkpoint_dir=output/train/mask_rcnn --eval_dir=output/eval/mask_rcnn
