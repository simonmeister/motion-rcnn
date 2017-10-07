# Tensorflow Object Detection API: Models.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "faster_rcnn_resnet_v1_feature_extractor",
    srcs = [
        "faster_rcnn_resnet_v1_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//tensorflow_models/slim:resnet_utils",
        "//tensorflow_models/slim:resnet_v1",
    ],
)

py_test(
    name = "faster_rcnn_resnet_v1_feature_extractor_test",
    srcs = [
        "faster_rcnn_resnet_v1_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_resnet_v1_feature_extractor",
        "//tensorflow",
    ],
)
