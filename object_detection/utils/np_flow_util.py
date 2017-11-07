# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import numpy as np


def gt_flow_and_mask(raw_flow):
  gt_flow_mask = np.float32((raw_flow[:, :, 0] * raw_flow[:, :, 1]) != np.nan)
  gt_flow_mask = np.expand_dims(gt_flow_mask, axis=2)
  gt_flow = np.nan_to_num(raw_flow)
  return gt_flow, gt_flow_mask
