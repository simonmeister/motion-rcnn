# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import tensorflow as tf


def clip_to_open_interval(x, xmin=-1.0, xmax=1.0, eps=1e-08):
  """Clip value to be strictly within the limits.
  E.g., the return value with default limits is safe to be
  used inside acos."""
  return tf.clip_by_value(x, xmin + eps, xmax - eps)


def euler_to_rot(x, y, z, sine_inputs=True):
    """Compose 3d rotations (in batches) from angles.
    Args:
      x, y, z: tensor of shape (N, 1)
      sine_inputs: if true, inputs are given as angle sines with
        values in [-1, 1], if false, as raw angles in radians.
    Returns:
      rotations: tensor of shape (N, 3, 3)
    """
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 1)
    z = tf.expand_dims(z, 1)

    if sine_inputs:
      sin_x = x
      sin_y = y
      sin_z = z
      eps = 1e-07
      cos_x = tf.sqrt(1 - tf.square(sin_x) + eps)
      cos_y = tf.sqrt(1 - tf.square(sin_y) + eps)
      cos_z = tf.sqrt(1 - tf.square(sin_z) + eps)
    else:
      sin_x = tf.sin(x)
      sin_y = tf.sin(y)
      sin_z = tf.sin(z)
      cos_x = tf.cos(x)
      cos_y = tf.cos(y)
      cos_z = tf.cos(z)

    zero = tf.zeros_like(sin_x)
    one = tf.ones_like(sin_x)

    rot_x_1 = tf.stack([one, zero, zero], axis=2)
    rot_x_2 = tf.stack([zero, cos_x, -sin_x], axis=2)
    rot_x_3 = tf.stack([zero, sin_x, cos_x], axis=2)
    rot_x = tf.concat([rot_x_1, rot_x_2, rot_x_3], axis=1)

    rot_y_1 = tf.stack([cos_y, zero, sin_y], axis=2)
    rot_y_2 = tf.stack([zero, one, zero], axis=2)
    rot_y_3 = tf.stack([-sin_y, zero, cos_y], axis=2)
    rot_y = tf.concat([rot_y_1, rot_y_2, rot_y_3], axis=1)

    rot_z_1 = tf.stack([cos_z, -sin_z, zero], axis=2)
    rot_z_2 = tf.stack([sin_z, cos_z, zero], axis=2)
    rot_z_3 = tf.stack([zero, zero, one], axis=2)
    rot_z = tf.concat([rot_z_1, rot_z_2, rot_z_3], axis=1)

    return rot_z @ rot_x @ rot_y


def motion_loss(pred, target, weights):
  """
  Args:
    pred: tensor of shape [batch_size, num_anchors, 9]
    target: tensor of shape [batch_size, num_anchors, 15]
    weights: tensor of shape [batch_size, num_anchors]
  Returns:
    loss: a tensor of shape [batch_size, num_anchors]
  """
  batch_size, num_anchors = tf.unstack(tf.shape(pred)[:2])

  l_angle, l_trans, l_pivot = _motion_losses(
      tf.reshape(pred, [-1, 9]),
      tf.reshape(target, [-1, 15]))

  loss = l_angle + l_trans + l_pivot
  return tf.reshape(loss, [batch_size, num_anchors]) * weights


def _motion_losses(pred, target):
  """
  Args:
    pred: tensor of shape [num_predictions, 9] containing predicted
      angle sines, translation and pivot
    target: tensor of shape [num_predictions, 15] containing
      target rotation matrix (flat), translation and pivot.
  Returns:
    losses: three-tuple of tensors of shape [num_predictions] representing the
      rotation, translation and pivot loss for each instance
  """
  def _smoothl1_loss(diff):
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    return tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        1)

  def _l1_loss(diff):
    return tf.reduce_sum(tf.abs(diff), 1)

  pred = postprocess_detection_motions(pred)
  rot = tf.reshape(pred[:, 0:9], [-1, 3, 3])
  trans = pred[:, 9:12]
  pivot = pred[:, 12:15]

  gt_rot = tf.reshape(target[:, 0:9], [-1, 3, 3])
  gt_trans = target[:, 9:12]
  gt_pivot = target[:, 12:15]

  #eye_rot = tf.eye(3, batch_shape=tf.shape(rot)[:1])
  #rot_T = tf.transpose(rot, [0, 2, 1])
  d_rot = tf.reshape(gt_rot - rot, [-1, 9])
  d_trans = gt_trans - trans
  d_pivot = gt_pivot - pivot

  l_angle = _smoothl1_loss(d_rot)
  l_trans = _smoothl1_loss(d_trans)
  l_pivot = _smoothl1_loss(d_pivot)
  return l_angle, l_trans, l_pivot


def postprocess_detection_motions(pred):
  """Convert predicted motions to use matrix representation for rotations.
  Restrict range of angle sines to [-1, 1]"""
  angle_sines = clip_to_open_interval(pred[:, 0:3])
  rot = euler_to_rot(angle_sines[:, 0], angle_sines[:, 1], angle_sines[:, 2])
  rot_flat = tf.reshape(rot, [-1, 9])
  return tf.concat([rot_flat, pred[:, 3:]], axis=1)


def postprocess_camera_motion(pred):
  return postprocess_detection_motions(tf.expand_dims(pred, 0))[0, :]


def camera_motion_loss(pred, target):
  """Compute loss between predicted and ground truth camera motion.
  Args:
    pred: tensor of shape [batch_size, 6] containing predicted
      angle sines and translation.
    target: tensor of shape [batch_size, 12] containing
      target rotation matrix and translation.
  Returns:
    losses: a scalar
  """
  batch_size = tf.unstack(tf.shape(pred))[0]
  mock_pivot = tf.zeros([batch_size, 3])
  err_angle, err_trans, _ = _motion_losses(
    tf.concat([pred, mock_pivot], axis=1),
    tf.concat([target, mock_pivot], axis=1))

  return err_angle + err_trans


def get_3D_coords(depth, camera_intrinsics):
  def _pixels_to_3d(x, y, d):
      x = tf.expand_dims(tf.expand_dims(x, 0), 3)
      y = tf.expand_dims(tf.expand_dims(y, 0), 3)
      f, x0, y0 = tf.unstack(camera_intrinsics)
      factor = d / f
      X = (x - x0) * factor
      Y = (y - y0) * factor
      Z = d
      return X, Y, Z

  num, height, width = tf.unstack(tf.shape(depth))[:3]
  ys = tf.cast(tf.range(height), tf.float32)
  xs = tf.cast(tf.range(width), tf.float32)
  x, y = tf.meshgrid(xs, ys)
  X, Y, Z = _pixels_to_3d(x, y, depth)
  XYZ = tf.concat([X, Y, Z], axis=3)
  return XYZ
