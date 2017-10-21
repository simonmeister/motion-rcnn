# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import tensorflow as tf


def euler_to_rot(sin_x, sin_y, sin_z):
    """Compose 3d rotations (in batches) from angle sines.
    Args:
      sin_{x, y, z}: tensor of shape (N, 1) with values in [-1, 1]
    Returns:
      rotations: tensor of shape (N, 3, 3)
    """
    zero = tf.zeros_like(sin_x)
    one = tf.ones_like(sin_x)

    cos_x = tf.sqrt(1 - sin_x ** 2)
    cos_y = tf.sqrt(1 - sin_y ** 2)
    cos_z = tf.sqrt(1 - sin_z ** 2)

    rot_x_1 = tf.stack([one, zero, zero], axis=2)
    rot_x_2 = tf.stack([zero, cos_x, -sin_x], axis=2)
    rot_x_3 = tf.stack([zero, sin_x, cos_gx], axis=2)
    rot_x = tf.concat([rot_x_1, rot_x_2, rot_x_3], axis=1)

    rot_y_1 = tf.stack([cos_y, zero, sin_y], axis=2)
    rot_y_2 = tf.stack([zero, one, zero], axis=2)
    rot_y_3 = tf.stack([-sin_y, zero, cos_y], axis=2)
    rot_y = tf.concat([rot_y_1, rot_y_2, rot_y_3], axis=1)

    rot_z_1 = tf.stack([cos_z, -sin_z, zero], axis=2)
    rot_z_2 = tf.stack([sin_z, cos_z, zero], axis=2)
    rot_z_3 = tf.stack([zero, zero, one], axis=2)
    rot_z = tf.concat([rot_z_1, rot_z_2, rot_z_3], axis=1)

    return tf.matmul(rot_z, tf.matmul(rot_x, rot_y))


def motion_losses(pred, gt):
  """
  Args:
    pred: tensor of shape [num_predictions, 9]
    gt: tensor of shape [num_predictions, 15]
  Returns:
    a tensor of shape [num_predictions] representing the loss
    for each instance
  """
  rot = euler_to_rot(pred[:, 0], pred[:, 1], pred[:, 2])
  trans = pred[:, 3:6]
  pivot = pred[:, 6:9]

  gt_rot = gt[:, 0:9]
  gt_trans = gt[:, 9:12]
  gt_pivot = gt[:, 12:15]

  rot_T = tf.transpose(rot, [0, 2, 1])
  d_rot = tf.matmul(rot_T, gt_rot)
  d_trans = tf.matmul(rot_T, gt_trans - trans)
  d_pivot = gt_pivot - pivot

  err_angle = tf.acos(tf.clip_by_value((tf.trace(d_rot) - 1) / 2, -1, 1))
  err_trans = tf.norm(d_trans, axis=1)
  err_pivot = tf.norm(d_pivot, axis=1)

  return err_angle + err_trans + err_pivot
