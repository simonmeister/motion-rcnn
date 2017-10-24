# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import tensorflow as tf


def euler_to_rot(sin_x, sin_y, sin_z):
    """Compose 3d rotations (in batches) from angles.
    Args:
      x, y, z: tensor of shape (N, 1) with values in [-1, 1]
    Returns:
      rotations: tensor of shape (N, 3, 3)
    """
    #x = tf.expand_dims(x, 1)
    #y = tf.expand_dims(y, 1)
    #z = tf.expand_dims(z, 1)

    #sin_x = tf.sin(x)
    #sin_y = tf.sin(y)
    #sin_z = tf.sin(z)
    sin_x = tf.expand_dims(sin_x, 1)
    sin_y = tf.expand_dims(sin_y, 1)
    sin_z = tf.expand_dims(sin_z, 1)

    zero = tf.zeros_like(sin_x)
    one = tf.ones_like(sin_x)

    cos_x = tf.sqrt(1 - sin_x ** 2)
    cos_y = tf.sqrt(1 - sin_y ** 2)
    cos_z = tf.sqrt(1 - sin_z ** 2)
    #cos_x = tf.cos(x)
    #cos_y = tf.cos(y)
    #cos_z = tf.cos(z)

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

  err_angle, err_trans, err_pivot = _motion_losses(
      tf.reshape(pred, [-1, 9]),
      tf.reshape(target, [-1, 15]))

  total_err = err_angle + err_trans + err_pivot
  return tf.reshape(total_err, [batch_size, num_anchors]) * weights


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
  pred = predicted_motion_angles_to_matrices(pred)
  rot = tf.reshape(pred[:, 0:9], [-1, 3, 3])
  trans = pred[:, 9:12]
  pivot = pred[:, 12:15]

  gt_rot = tf.reshape(target[:, 0:9], [-1, 3, 3])
  gt_trans = target[:, 9:12]
  gt_pivot = target[:, 12:15]

  rot_T = tf.transpose(rot, [0, 2, 1])
  d_rot = rot_T @ gt_rot
  d_trans = tf.squeeze(rot_T @ tf.reshape(gt_trans - trans, [-1, 3, 1]))
  d_pivot = gt_pivot - pivot

  err_angle = tf.acos(tf.clip_by_value((tf.trace(d_rot) - 1) / 2, -1, 1))
  err_trans = tf.norm(d_trans, axis=1)
  err_pivot = tf.norm(d_pivot, axis=1)

  return err_angle, err_trans, err_pivot


def predicted_motion_angles_to_matrices(pred):
  """Convert predicted motions to use matrix representation for rotations.
  Also restrict range of angle sines to [-1, 1]"""
  #angle_sines = tf.maximum(tf.minimum(pred[:, 0:3], 2), 0) - 1
  angle_sines = tf.clip_by_value(pred[:, 0:3], -1, 1)
  rot = euler_to_rot(angle_sines[:, 0], angle_sines[:, 1], angle_sines[:, 2])
  rot_flat = tf.reshape(rot, [-1, 9])
  return tf.concat([rot_flat, pred[:, 3:]], axis=1)
