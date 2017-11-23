# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import numpy as np
import tensorflow as tf

from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_flow_util
from object_detection.utils import flow_util


def _pixels_to_3d(x, y, d, camera_intrinsics):
  f, x0, y0 = camera_intrinsics
  factor = d / f
  X = (x - x0) * factor
  Y = (y - y0) * factor
  Z = d
  return X, Y, Z


def _3d_to_pixels(points, camera_intrinsics):
  f, x0, y0 = camera_intrinsics
  X = points[:, :, 0]
  Y = points[:, :, 1]
  Z = points[:, :, 2]
  x = f * X / Z + x0
  y = f * Y / Z + y0
  return x, y


def dense_flow_from_motion(depth, motions, masks, camera_motion,
                           camera_intrinsics):
  """Compute optical flow map from depth and motion data.

  Args:
    depth: array with shape [height, width, 1].
    motions: array with shape [num_detections, 16],
      (rot_9, trans_3, piv_3, moving_1).
    masks: array with shape [num_detections, height, width]
    camera_motion: array with shape [12].
    camera_intrinsics: array with shape [3].

  Returns:
    Array with shape [height, width, 2] representing the optical flow
    in x and y directions.
  """
  h, w = depth.shape[:2]
  depth = depth[:, :, 0]
  x_range = np.linspace(0, w - 1, w)
  y_range = np.linspace(0, h - 1, h)

  x, y = np.meshgrid(x_range, y_range)
  X, Y, Z = _pixels_to_3d(x, y, depth, camera_intrinsics)
  points = np.stack([x, y], axis=2)
  P = np.stack([X, Y, Z], axis=2)

  for i in range(motions.shape[0]):
    moving = motions[i, 15]
    if moving < 0.5:
      continue
    rot = np.reshape(motions[i, :9], [3, 3])
    trans = np.reshape(motions[i, 9:12], [3])
    pivot = np.reshape(motions[i, 12:15], [3])
    mask = np.expand_dims(masks[i, :, :], 2)
    P += mask * ((P - pivot).dot(rot.T) + pivot + trans - P)

  #moving_cam = camera_motion[12]
  rot_cam = np.reshape(camera_motion[:9], [3, 3])
  trans_cam = np.reshape(camera_motion[9:12], [-1])
  #if moving_cam > 0.5:
  P = P.dot(rot_cam.T) + trans_cam

  x_t, y_t = _3d_to_pixels(P, camera_intrinsics)
  points_t = np.stack([x_t, y_t], axis=2)

  flow = points_t - points
  return flow.astype(np.float32)


def evaluate_optical_flow(depth, motions, masks, camera_motion,
                          camera_intrinsics, gt_flow):
  """Note: currently, calls can only be made inside a tf.Session."""
  flow = dense_flow_from_motion(
      depth, motions, masks, camera_motion, camera_intrinsics)
  gt_flow, gt_flow_mask = np_flow_util.gt_flow_and_mask(gt_flow)

  # to tensorflow
  flow = tf.expand_dims(flow, 0)
  gt_flow = tf.expand_dims(gt_flow, 0)
  gt_flow_mask = tf.expand_dims(gt_flow_mask, 0)
  avg = flow_util.flow_error_avg(gt_flow, flow, gt_flow_mask).eval()
  outliers = flow_util.outlier_ratio(gt_flow, flow, gt_flow_mask).eval()

  error_dict = {'AEE': avg.item(), 'Fl-all': outliers.item()}
  return error_dict


def euler_to_rot(x, y, z):
  rot_x = np.array([[1, 0, 0],
                    [0, np.cos(x), -np.sin(x)],
                    [0, np.sin(x), np.cos(x)]],
                   dtype=np.float32)
  rot_z = np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]],
                   dtype=np.float32)
  rot_y = np.array([[np.cos(y), 0, np.sin(y)],
                    [0, 1, 0],
                    [-np.sin(y), 0, np.cos(y)]],
                   dtype=np.float32)
  return rot_z @ rot_x @ rot_y


def _rotation_angle(mat):
  return np.arccos(np.clip((np.trace(mat, axis1=1, axis2=2) - 1) / 2, -1, 1))


def _get_rotation_eye(rot):
  single_eye = np.eye(3)
  return np.tile(np.expand_dims(single_eye, 0), [rot.shape[0], 1, 1])


def _motion_errors(pred, target, has_moving=True):
  """
  Args:
    pred: array of shape [num_predictions, 15] containing predicted
      rotation matrix, translation and pivot
    target: array of shape [num_predictions, 15] containing
      target rotation matrix, translation and pivot.
  Returns:
    error_dict: dictionary of floats representing the mean
      rotation, translation, pivot, relative rotation and relative translation
      errors
  """
  rot = np.reshape(pred[:, 0:9], [-1, 3, 3])
  trans = pred[:, 9:12]
  pivot = pred[:, 12:15]

  if has_moving:
    moving = pred[:, 15:16] > 0.5
    rot = np.where(np.expand_dims(moving, 2), rot, _get_rotation_eye(rot))
    trans = np.where(moving, trans, np.zeros_like(trans))
    gt_moving = target[:, 15:16] > 0.5
    TP = np.sum(np.logical_and(gt_moving == 1, moving == 1))
    FP = np.sum(np.logical_and(gt_moving == 0, moving == 1))
    FN = np.sum(np.logical_and(gt_moving == 1, moving == 0))
    moving_dict = {
      'moving_precision': TP / (TP + FP) if np.asscalar(TP + FP) > 0 else 1,
      'moving_recall': TP / (TP + FN) if np.asscalar(TP + FN) > 0 else 1
    }
  else:
    moving_dict = {}

  gt_rot = np.reshape(target[:, 0:9], [-1, 3, 3])
  gt_trans = target[:, 9:12]
  gt_pivot = target[:, 12:15]

  rot_T = np.transpose(rot, [0, 2, 1])
  d_rot = gt_rot @ rot_T
  d_trans = gt_trans - trans
  d_pivot = gt_pivot - pivot

  err_angle = _rotation_angle(d_rot)
  err_trans = np.linalg.norm(d_trans, axis=1)
  err_pivot = np.linalg.norm(d_pivot, axis=1)

  np.seterr(divide='ignore')
  err_rel_angle = err_angle / _rotation_angle(gt_rot)
  err_rel_trans = err_trans / np.linalg.norm(gt_trans, axis=1)
  err_rel_angle = err_rel_angle[np.isfinite(err_rel_angle)]
  err_rel_trans = err_rel_trans[np.isfinite(err_rel_trans)]

  mean_angle = np.mean(np.degrees(err_angle))
  mean_trans = np.mean(err_trans)
  mean_pivot = np.mean(err_pivot)

  if len(err_rel_angle) > 0:
    mean_rel_angle = np.mean(err_rel_angle)
  else:
    mean_rel_angle = np.array([0])

  if len(err_rel_trans) > 0:
    mean_rel_trans = np.mean(err_rel_trans)
  else:
    mean_rel_trans = np.array([0])

  error_dict = {
      'mAngle': mean_angle,
      'mTrans': mean_trans,
      'mPivot': mean_pivot,
      'mRelAngle': mean_rel_angle,
      'mRelTrans': mean_rel_trans,
      'mAveAngle': np.mean(_rotation_angle(gt_rot)),
      'mAveTrans': np.mean(np.linalg.norm(gt_trans, axis=1))}

  error_dict = {k: np.asscalar(v) for (k, v) in error_dict.items()}
  error_dict.update(moving_dict)
  return error_dict


def evaluate_instance_motions(gt_boxes,
                              gt_motions,
                              detected_boxes,
                              detected_motions,
                              matching_iou_threshold=.5):
  gt_boxlist = np_box_list.BoxList(gt_boxes)
  detected_boxlist = np_box_list.BoxList(detected_boxes)

  iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
  max_overlap_gt_ids = np.argmax(iou, axis=1)

  pred_list = []
  target_list = []
  for i in range(detected_boxlist.num_boxes()):
    gt_id = max_overlap_gt_ids[i]
    if iou[i, gt_id] >= matching_iou_threshold:
      pred_list.append(detected_motions[i, :])
      target_list.append(gt_motions[gt_id, :])
  if len(pred_list) == 0:
    return {
        'mAngle': 0,
        'mTrans': 0,
        'mPivot': 0,
        'mRelAngle': 0,
        'mRelTrans': 0,
        'mAveAngle': 0,
        'mAveTrans': 0}
  pred = np.stack(pred_list, axis=0)
  target = np.stack(target_list, axis=0)
  return _motion_errors(pred, target)


def evaluate_camera_motion(pred, target):
  mock_pivot = np.zeros([1, 3])
  error_dict = _motion_errors(
    np.concatenate([np.expand_dims(pred, 0), mock_pivot], axis=1),
    np.concatenate([np.expand_dims(target, 0), mock_pivot], axis=1),
    has_moving=False)
  del error_dict['mPivot']
  return error_dict
