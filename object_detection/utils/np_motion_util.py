# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import numpy as np


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
    motions: array with shape [num_detections, 15].
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
    rot = np.reshape(motions[i, :9], [3, 3])
    trans = np.reshape(motions[i, 9:12], [3])
    pivot = np.reshape(motions[i, 12:], [3])
    mask = np.expand_dims(masks[i, :, :], 2)
    P += mask * ((P - pivot).dot(rot.T) + pivot + trans - P)

  rot_cam = np.reshape(camera_motion[:9], [3, 3])
  trans_cam = np.reshape(camera_motion[9:], [-1])
  P = P.dot(rot_cam.T) + trans_cam

  x_t, y_t = _3d_to_pixels(P, camera_intrinsics)
  points_t = np.stack([x_t, y_t], axis=2)

  flow = points_t - points
  return flow.astype(np.float32)


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
