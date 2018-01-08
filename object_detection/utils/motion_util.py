# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import tensorflow as tf
import numpy as np


def clip_to_open_interval(x, xmin=-1.0, xmax=1.0, eps=1e-08):
  """Clip value to be strictly within the limits.
  E.g., the return value with default limits is safe to be
  used inside acos."""
  return tf.clip_by_value(x, xmin + eps, xmax - eps)


def _smoothl1_loss(diff, reduce_dims=[1]):
  abs_diff = tf.abs(diff)
  abs_diff_lt_1 = tf.less(abs_diff, 1)
  return tf.reduce_sum(
      tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
      reduce_dims)


def _l1_loss(diff, reduce_dims=[1]):
  return tf.reduce_sum(tf.abs(diff), reduce_dims)


def _motion_losses(pred, target, has_moving=True, has_pivot=True):
  """
  Args:
    pred: tensor of shape [num_predictions, num_pred] containing predicted
      rotation matrix, translation, pivot (optional) and moving logits
      (optional).
    target: tensor of shape [num_predictions, num_gt] containing
      target rotation matrix (flat), translation, pivot (optional)
      and moving flag.
  Returns:
    losses: three-tuple of tensors of shape [num_predictions] representing the
      rotation, translation and pivot loss for each instance
  """
  num_pred = int(has_moving) * 2 + int(has_pivot) * 3 + 7
  num_gt = 1 + int(has_pivot) * 3 + 7
  assert_pred = tf.assert_equal(tf.shape(pred)[1], num_pred,
                                name='motion_loss_assert_pred')
  assert_target = tf.assert_equal(tf.shape(target)[1], num_gt,
                                  name='motion_loss_assert_target')
  with tf.control_dependencies([assert_pred, assert_target]):
    q = pred[:, :4]
    trans = pred[:, 4:7]

    gt_q = target[:, :4]
    gt_trans = target[:, 4:7]

    d_q = gt_q - q
    d_trans = gt_trans - trans

    l_angle = _smoothl1_loss(d_q)
    l_trans = _smoothl1_loss(d_trans)

    if has_pivot:
      pivot = pred[:, 7:10]
      gt_pivot = target[:, 7:10]
      d_pivot = gt_pivot - pivot
      l_pivot = _smoothl1_loss(d_pivot)
    else:
      l_pivot = None

    if has_moving:
      moving = pred[:, 10:12]
      gt_moving = target[:, 10]
      l_moving = tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.one_hot(tf.cast(gt_moving, dtype=tf.int32),
                            depth=2, dtype=tf.float32),
          logits=moving)
    else:
      l_moving = None,
      gt_moving = None

  l_angle = l_angle * 100
  return l_angle, l_trans, l_pivot, l_moving, gt_moving


def batch_postprocess_motions(pred, has_pivot=True, has_moving=True,
                              keep_logits=False):
  """Variant of postprocess_motions with two outer dimensions.

  Args:
    pred: tensor of shape [batch_size, num_boxes, num_params],
      where num_params is 7 + 2 * has_moving + 3 * has_pivot.

  Returns:
    processed: tensor of shape [batch_size, num_boxes, num_params_processed],
      where num_params_processed = num_params + 6.
  """
  batch_size, num_boxes, num_params = tf.unstack(tf.shape(pred))
  pred = tf.reshape(pred, [-1, num_params])
  res = postprocess_motions(pred, has_pivot, has_moving, keep_logits)
  res = tf.reshape(res, [batch_size, num_boxes, -1])
  return res


def postprocess_motions(pred,
                        has_pivot=True,
                        has_moving=True,
                        keep_logits=True):
  """Convert predicted motions to use matrix representation for rotations.
  Restrict range of angle sines to [-1, 1].
  If keep_logits=False, convert moving logits to scores.

  By convention,
  * the first 4 entries (along dim 1) of pred correspond to the orientation quaternion,
  * the next 3 entries correspond to the translation
  * (optional) next, there are 3 entries for the pivot if has_pivot=True
  * (optional) next, 2 entries (logits for not-moving and moving class)
    if has_moving=True.

  Args:
    pred: tensor of shape [num_boxes, num_pred].

  Returns:
    processed: tensor of shape [num_boxes, num_out],
      where num_out = num_pred + 3.
  """
  #num_pred = int(has_moving) * 2 + int(has_pivot) * 3 + 7
  #assert_pred = tf.assert_equal(tf.shape(pred)[1], num_pred,
  #                              name='postprocess_motions_assert_pred')
  #with tf.control_dependencies([assert_pred]):
  q = pred[:, :4] # * 1e-4
  w, x, y, z = tf.split(q, 4, axis=-1)
  # the initial (and zero) prediction should be the identity rotation (1, 0, 0, 0)
  w = 1 - w
  q = tf.concat([w, x, y, z], axis=-1)
  q = q / tf.maximum(
      tf.norm(q, ord='euclidean', keep_dims=True, axis=1), 1e-12)
  res = q
  trans = pred[:, 4:7]
  res = tf.concat([res, trans], axis=1)
  if has_pivot:
    pivot = pred[:, 7:10]
    res = tf.concat([res, pivot], axis=1)
    moving_start = 10
  else:
    moving_start = 7
  if has_moving:
    moving = pred[:, moving_start:moving_start+2]
    if not keep_logits:
      moving_score = tf.nn.softmax(moving)[:, 1:2]
      moving = tf.cast(moving_score > 0.5, dtype=tf.float32)
    res = tf.concat([res, moving], axis=1)
  return res


def postprocess_detection_motions(pred, has_moving=True, keep_logits=True):
  """Postprocess instance motions."""
  return postprocess_motions(pred, has_pivot=True, has_moving=has_moving,
                             keep_logits=keep_logits)


def postprocess_camera_motion(pred):
  return postprocess_motions(pred, has_pivot=False, has_moving=False)


def motion_loss(pred, target, weights):
  """
  Args:
    pred: tensor of shape [batch_size, num_anchors, 12]
    target: tensor of shape [batch_size, num_anchors, 11]
    weights: tensor of shape [batch_size, num_anchors]
  Returns:
    loss: a tensor of shape [batch_size, num_anchors]
  """
  batch_size, num_anchors = tf.unstack(tf.shape(pred)[:2])

  l_angle, l_trans, l_pivot, l_moving, gt_moving = _motion_losses(
      postprocess_detection_motions(tf.reshape(pred, [-1, 12]),
                                    keep_logits=True),
      tf.reshape(target, [-1, 11]),
      has_moving=True,
      has_pivot=True)

  loss = (l_angle + l_trans) * gt_moving + l_pivot + l_moving
  return tf.reshape(loss, [batch_size, num_anchors]) * weights


def camera_motion_loss(pred, target):
  """Compute loss between predicted and ground truth camera motion.
  Args:
    pred: tensor of shape [batch_size, 7] containing predicted
      rotation and translation.
    target: tensor of shape [batch_size, 7] containing
      target rotation and translation.
  Returns:
    losses: a scalar
  """
  l_angle, l_trans, _, _, _ = _motion_losses(
    postprocess_camera_motion(pred),
    target,
    has_moving=False,
    has_pivot=False)

  return l_angle + l_trans


# Flow loss
#

def q_multiply(q1, q2):
  w1, x1, y1, z1 = tf.split(q1, 4, axis=-1)
  w2, x2, y2, z2 = tf.split(q2, 4, axis=-1)
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
  z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
  return tf.concat((w, x, y, z), axis=-1)


def q_conjugate(q):
  w, x, y, z = tf.split(q, 4, axis=-1)
  return tf.concat([w, -x, -y, -z], axis=-1)


def q_rotate(q, p):
  p = tf.concat([tf.zeros(tf.unstack(tf.shape(p))[:-1] + [1]), p], axis=-1)
  return q_multiply(q_multiply(q, p), q_conjugate(q))[..., 1:]


def _pixels_to_3d(positions, d, camera_intrinsics):
  x, y = tf.split(positions, 2, axis=-1)
  f, x0, y0 = tf.unstack(camera_intrinsics)
  factor = d / f
  X = (x - x0) * factor
  Y = (y - y0) * factor
  Z = d
  points = tf.concat([X, Y, Z], axis=-1)
  return points


def _3d_to_pixels(points, camera_intrinsics):
  X, Y, Z = tf.split(points, 3, axis=-1)
  f, x0, y0 = tf.unstack(camera_intrinsics)
  x = f * X / Z + x0
  y = f * Y / Z + y0
  positions = tf.concat([x, y], axis=-1)
  return positions


def get_2D_coords(height, width):
  ys = tf.cast(tf.range(height), tf.float32)
  xs = tf.cast(tf.range(width), tf.float32)
  x, y = tf.meshgrid(xs, ys)
  x = tf.expand_dims(tf.expand_dims(x, 0), 3)
  y = tf.expand_dims(tf.expand_dims(y, 0), 3)
  return tf.concat([x, y], axis=-1)


def get_3D_coords(depth, camera_intrinsics):
  num, height, width = tf.unstack(tf.shape(depth))[:3]
  positions = get_2D_coords(height, width)
  return _pixels_to_3d(positions, depth, camera_intrinsics)


def flow_camera_motion_loss(gt_masks, camera_motion, depth, flow, camera_intrinsics):
  """Supervise camera_motion with optical flow.
  Args:
    gt_masks: tensor of shape [batch_size, num_boxes, image_height, image_width]
    camera_motion: tensor of shape [batch_size, 7]
    depth: tensor of shape [batch_size, image_height, image_width, 1]
      containing predicted or ground truth depth.
    flow: tensor of shape [batch_size, image_height, image_width, 2]
      containing ground truth optical flow.
    camera_intrinsics: tensor of shape [batch_size, 3]
  Returns:
    loss: scalar
  """
  num_batch, height, width = tf.unstack(tf.shape(flow))[:3]
  static_pixel_mask = tf.expand_dims(
      tf.reduce_prod(1 - gt_masks, axis=1), axis=3)

  positions = get_2D_coords(height, width)
  positions = tf.tile(positions, [num_batch, 1, 1, 1])
  points = _pixels_to_3d(positions, depth, camera_intrinsics)

  points_t = _apply_camera_motion(points, camera_motion)
  positions_t = _3d_to_pixels(points_t, camera_intrinsics)
  reprojection_flow = positions_t - positions

  normalizer = tf.reduce_sum(static_pixel_mask, axis=[1, 2, 3])
  loss = _l1_loss(
      (reprojection_flow - flow) * static_pixel_mask,
      [1, 2, 3]) / normalizer
  return loss


def flow_motion_loss(boxes, masks, motions, camera_motion,
                     depth, flow, camera_intrinsics, weights):
  """Supervise motion with optical flow.
  Args:
    boxes: tensor of shape [batch_size, num_boxes, 4]
    masks: tensor of shape [batch_size, num_boxes, mask_height, mask_width]
    motions: tensor of shape [batch_size, num_boxes, 11]
    camera_motion: tensor of shape [batch_size, 7]
    depth: tensor of shape [batch_size, image_height, image_width, 1]
      containing predicted or ground truth depth.
    camera_intrinsics: tensor of shape [batch_size, 3]
  Returns:
    loss: scalar
  """
  num_batch, num_boxes, mask_height, mask_width = tf.unstack(tf.shape(masks))[:4]
  masks = tf.expand_dims(masks, axis=4)

  boxes_flat = tf.reshape(boxes, [-1, 4])
  batch_indices = tf.reshape(
      tf.tile(tf.expand_dims(tf.range(num_batch), 0), [num_boxes, 1]),
      [-1])

  # TODO this will lead to divbyzero if we crop boxes going over image boundaries
  # TODO we also have to mask these pixels in the losses! how about the mask reg loss in that case??
  #
  d_flat = tf.image.crop_and_resize(
      image=depth,
      boxes=boxes_flat,
      box_ind=batch_indices,
      crop_size=[mask_height, mask_width])
  d = tf.reshape(
      d_flat,
      [num_batch, num_boxes, mask_height, mask_width, 1])

  flow_crops_flat = tf.image.crop_and_resize(
      image=flow,
      boxes=boxes_flat,
      box_ind=batch_indices,
      crop_size=[mask_height, mask_width])
  flow_crops = tf.reshape(
      flow_crops_flat,
      [num_batch, num_boxes, mask_height, mask_width, 2])

  def _py_create_2d_grids(np_boxes, height, width):
    pos_grids = []
    num_batch, num_boxes = np_boxes.shape[:2]
    np_boxes_flat = np.reshape(np_boxes, [-1, 4])
    for i in range(np_boxes_flat.shape[0]):
      y0, x0, y1, x1 = np_boxes_flat[i, :]
      y0 *= np.round(height - 1)
      y1 *= np.round(height - 1)
      x0 *= np.round(width - 1)
      x1 *= np.round(width - 1)
      ys = np.linspace(y0, y1, num=height, dtype=np.float32)
      xs = np.linspace(x0, x1, num=width, dtype=np.float32)
      x_grid, y_grid = np.meshgrid(xs, ys)
      pos_grids.append(np.stack([x_grid, y_grid], axis=2))
    pos_flat = np.stack(pos_grids, axis=0)
    pos = np.reshape(pos_flat, [num_batch, num_boxes, height, width, 2])
    return pos

  positions = tf.py_func(
      _py_create_2d_grids,
      [boxes, mask_height, mask_width],
      tf.float32)

  points = _pixels_to_3d(positions, d, camera_intrinsics)

  # make trailing dimensions of points compatible with motions
  # [batch_size, num_boxes, h, w, 3] -> [h, w, batch_size, num_boxes, 3]
  points = tf.transpose(points, perm=[2, 3, 0, 1, 4])
  masks = tf.transpose(masks, perm=[2, 3, 0, 1, 4])

  points_t_obj = _apply_object_motions(points, motions, masks)

  # make trailing dimensions of points compatible with camera motions
  # [h, w, batch_size, num_boxes, 3]-> [h, w, num_boxes, batch_size, 3]
  points_t_obj = tf.transpose(points_t_obj, perm=[0, 1, 3, 2, 4])
  points_t = _apply_camera_motion(points_t_obj, camera_motion)

  # switch back to [batch_size, num_boxes, h, w, 3]
  points_t = tf.transpose(points_t, perm=[3, 2, 0, 1, 4])

  positions_t = _3d_to_pixels(points_t, camera_intrinsics)
  reprojection_flow = positions_t - positions

  normalizer = tf.to_float(mask_height * mask_width)
  loss = _l1_loss(
      reprojection_flow - flow_crops,
      [2, 3, 4]) / normalizer
  return loss * weights


def _apply_object_motions(points, motions, masks):
  """Transform points with per-object motions, weighted by per-pixel object masks.
  Args:
    points: tensor of shape [mask_height, mask_width, batch_size, num_boxes, 3]
    motions: tensor of shape [batch_size, num_boxes, 9]
    masks: tensor of shape [mask_height, mask_width, batch_size, num_boxes, 1]
  returns:
    points_t: tensor of same shape as 'points'
  """
  motions = batch_postprocess_motions(motions, has_pivot=True, keep_logits=False,
                                      has_moving=True)
  q = motions[:, :, :4]
  trans = motions[:, :, 4:7]
  pivot = motions[:, :, 7:10]
  moving = motions[:, :, 10:11]

  points_t_all = q_rotate(q, points - pivot) + trans + pivot

  # compute difference between points and transformed points to obtain increments
  # which we can apply to the original points
  # only transform points where mask is 1 and where the object is moving
  diffs = points_t_all - points
  points_t = points + ((moving * masks) * diffs)

  return points_t


def _apply_camera_motion(points, motions):
  """Transform all points with global camera motion.
  Args:
    points: tensor of shape [..., batch_size, 3] # mask_height, mask_width, num_boxes
    motions: tensor of shape [batch_size, 7]
  returns:
    points_t: tensor of same shape as 'points'
  """
  motions = postprocess_motions(motions, has_pivot=False, has_moving=False)
  q = motions[:, :4]
  trans = motions[:, 4:7]

  return q_rotate(q, points) + trans
