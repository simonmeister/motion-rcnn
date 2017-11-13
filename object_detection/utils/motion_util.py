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
  num_pred = int(has_moving) * 2 + int(has_pivot) * 3 + 12
  num_gt = 1 + int(has_pivot) * 3 + 12
  assert_pred = tf.assert_equal(tf.shape(pred)[1], num_pred,
                                name='_motion_loss_assert_pred')
  assert_target = tf.assert_equal(tf.shape(target)[1], num_gt,
                                  name='_motion_loss_assert_target')
  with tf.control_dependencies([assert_pred, assert_target]):
    rot = tf.reshape(pred[:, 0:9], [-1, 3, 3])
    trans = pred[:, 9:12]

    gt_rot = tf.reshape(target[:, 0:9], [-1, 3, 3])
    gt_trans = target[:, 9:12]

    d_rot = tf.reshape(gt_rot - rot, [-1, 9])
    d_trans = gt_trans - trans

    l_angle = _smoothl1_loss(d_rot)
    l_trans = _smoothl1_loss(d_trans)

    if has_pivot:
      pivot = pred[:, 12:15]
      gt_pivot = target[:, 12:15]
      d_pivot = gt_pivot - pivot
      l_pivot = _smoothl1_loss(d_pivot)
    else:
      l_pivot = None

    if has_moving:
      moving = pred[:, 15:17]
      gt_moving = target[:, 15]
      l_moving = tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.one_hot(tf.cast(gt_moving, dtype=tf.int32),
                            depth=2, dtype=tf.float32),
          logits=moving)
    else:
      l_moving = None,
      gt_moving = None
  return l_angle, l_trans, l_pivot, l_moving, gt_moving


def batch_postprocess_motions(pred, has_pivot=True, has_moving=True,
                              keep_logits=False):
  """Variant of postprocess_motions with two outer dimensions.

  Args:
    pred: tensor of shape [batch_size, num_boxes, num_params],
      where num_params is 6 + 2 * has_moving + 3 * has_pivot.

  Returns:
    processed: tensor of shape [batch_size, num_boxes, num_params_processed],
      where num_params_processed = num_params + 6.
  """
  batch_size, num_boxes, num_params = tf.unstack(tf.shape(pred))
  pred = tf.reshape(pred, [-1, num_params])
  res = postprocess_motions(pred, has_pivot, has_moving, keep_logits)
  res = tf.reshape(res, [batch_size, num_boxes, -1])
  return res


def postprocess_motions(pred, has_pivot=True, has_moving=True,
                        keep_logits=True):
  """Convert predicted motions to use matrix representation for rotations.
  Restrict range of angle sines to [-1, 1].
  If keep_logits=False, convert moving logits to scores.

  By convention,
  * the first 3 entries (along dim 1) of pred are the 3 angle sines,
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
  num_pred = int(has_moving) * 2 + int(has_pivot) * 3 + 6
  assert_pred = tf.assert_equal(tf.shape(pred)[1], num_pred,
                                name='_postprocess_motions_assert_pred')
  with tf.control_dependencies([assert_pred]):
    angle_sines = clip_to_open_interval(pred[:, 0:3])
    rot = euler_to_rot(angle_sines[:, 0], angle_sines[:, 1], angle_sines[:, 2])
    res = tf.reshape(rot, [-1, 9])
    trans = pred[:, 3:6]
    res = tf.concat([res, trans], axis=1)
    if has_pivot:
      pivot = pred[:, 6:9]
      res = tf.concat([res, pivot], axis=1)
      moving_start = 9
    else:
      moving_start = 6
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
    pred: tensor of shape [batch_size, num_anchors, 9]
    target: tensor of shape [batch_size, num_anchors, 15]
    weights: tensor of shape [batch_size, num_anchors]
  Returns:
    loss: a tensor of shape [batch_size, num_anchors]
  """
  batch_size, num_anchors = tf.unstack(tf.shape(pred)[:2])

  l_angle, l_trans, l_pivot, l_moving, gt_moving = _motion_losses(
      postprocess_detection_motions(tf.reshape(pred, [-1, 11]),
                                    keep_logits=True),
      tf.reshape(target, [-1, 16]),
      has_moving=True,
      has_pivot=True)

  loss = (l_angle + l_trans) * gt_moving + l_pivot + l_moving
  return tf.reshape(loss, [batch_size, num_anchors]) * weights


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
  l_angle, l_trans, _, _, _ = _motion_losses(
    postprocess_camera_motion(pred),
    target,
    has_moving=False,
    has_pivot=False)

  return l_angle + l_trans


def _pixels_to_3d(x, y, d, camera_intrinsics):
  f, x0, y0 = tf.unstack(camera_intrinsics)
  factor = d / f
  X = (x - x0) * factor
  Y = (y - y0) * factor
  Z = d
  return X, Y, Z


def _3d_to_pixels(X, Y, Z, camera_intrinsics):
  f, x0, y0 = tf.unstack(camera_intrinsics)
  x = f * X / Z + x0
  y = f * Y / Z + y0
  return x, y


def get_3D_coords(depth, camera_intrinsics):
  num, height, width = tf.unstack(tf.shape(depth))[:3]
  ys = tf.cast(tf.range(height), tf.float32)
  xs = tf.cast(tf.range(width), tf.float32)
  x, y = tf.meshgrid(xs, ys)
  x = tf.expand_dims(tf.expand_dims(x, 0), 3)
  y = tf.expand_dims(tf.expand_dims(y, 0), 3)
  X, Y, Z = _pixels_to_3d(x, y, depth, camera_intrinsics)
  XYZ = tf.concat([X, Y, Z], axis=3)
  return XYZ


def flow_motion_loss(boxes, masks, motions, camera_motion,
                     depth, flow, camera_intrinsics, weights):
  """Supervise motion with optical flow.
  Args:
    boxes: tensor of shape [batch_size, num_boxes, 4]
    masks: tensor of shape [batch_size, num_boxes, mask_height, mask_width]
    camera_motion: tensor of shape [batch_size, 9]
    depth: tensor of shape [batch_size, image_height, image_width, 1]
      containing predicted or ground truth depth.
    flow: tensor of shape [batch_size, image_height, image_width, 2]
      containing ground truth optical flow.
    camera_intrinsics: tensor of shape [batch_size, 3]
  Returns:
    loss: scalar
  """
  num_batch, num_boxes, mask_height, mask_width = tf.shape(masks)[:4]
  masks = tf.expand_dims(masks, axis=4)

  boxes_flat = tf.reshape(boxes, [-1, 4])
  batch_indices = tf.reshape(
      tf.tile(tf.expand_dims(tf.range(num_batch), 0), [num_boxes, 1]),
      [-1])

  d_flat = tf.image.crop_and_resize(
      image=depth,
      boxes=boxes_flat,
      box_ind=batch_indices,
      crop_size=[mask_height, mask_width])
  d = tf.reshape(
      d_flat,
      [num_batch, num_boxes, mask_height, mask_width])

  flow_crops_flat = tf.image.crop_and_resize(
      image=flow,
      boxes=boxes,
      box_ind=batch_indices,
      crop_size=[mask_height, mask_width])
  flow_crops = tf.reshape(
      flow_crops_flat,
      [num_batch, num_boxes, mask_height, mask_width, -1])

  def _py_create_2d_grids(np_boxes, height, width):
    x_grids = []
    y_grids = []
    num_batch, num_boxes = np_boxes.shape[:2]
    np_boxes_flat = np.reshape(np_boxes, [-1, 4])
    for i in np_boxes_flat.shape[0]:
      y0, x0, y1, x1 = np_boxes_flat[i]
      ys = np.linspace(y0, y1, num=height, dtype=np.float32)
      xs = np.linspace(x0, x1, num=width, dtype=np.float32)
      x_grid, y_grid = np.meshgrid(xs, ys)
      x_grids.append(x_grid)
      y_grids.append(y_grid)
    x_flat = np.stack(x_grids, axis=1)
    y_flat = np.stack(y_grids, axis=1)
    x = np.reshape(x_flat, [num_batch, num_boxes, height, width])
    y = np.reshape(y_flat, [num_batch, num_boxes, height, width])
    return x, y

  x, y = tf.py_func(
      py_crop,
      [boxes, mask_height, mask_width],
      [tf.float32, tf.float32])

  # point cloud of shape [batch_size, num_boxes, mask_height, mask_width, 3]
  X, Y, Z = _pixels_to_3d(x, y, d, camera_intrinsics)
  points = tf.stack([X, Y, Z], axis=4)

  # make trailing dimensions of points compatible with motions
  # [batch_size, num_boxes, h, w, 3] -> [h, w, batch_size, num_boxes, 3]
  points = tf.transpose(points, perm=[2, 3, 0, 1, 4])
  masks = tf.transpose(masks, perm=[2, 3, 0, 1])

  points_t_obj = _apply_object_motions(points, motions, masks)

  # make trailing dimensions of points compatible with camera motions
  # [h, w, batch_size, num_boxes, 3]-> [h, w, num_boxes, batch_size, 3]
  points_t_obj = tf.transpose(points_t_obj, perm=[0, 1, 2, 3, 4])
  points_t = _apply_camera_motion(points_t_obj, camera_motion)

  # switch back to [batch_size, num_boxes, h, w, 3]
  points_t = tf.transpose(points_t, perm=[3, 2, 0, 1, 4])

  x_t, y_t = _3d_to_pixels(*tf.unstack(points_t, axis=4, num=3),
                           camera_intrinsics)
  positions_t = tf.stack([x_t, y_t], axis=4)
  positions = tf.stack([x, y], axis=4)
  reprojection_flow = positions_t - positions

  normalizer = mask_height * mask_width
  loss = _smoothl1_loss(
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
  rot = tf.reshape(motions[:, :, 0:9], [-1, 3, 3])
  trans = motions[:, :, 9:12]
  pivot = motions[:, :, 12:15]
  moving = motions[:, :, 15:17]

  # broadcast subtract pivot point to center points at object coord systems
  points_centered = points - pivots

  # rotate the centered points with the rotation matrix of each object,
  # [batch_size, num_boxes, 3, 3], [h, w, batch_size, num_boxes, 3] ->
  # [h, w, batch_size, num_boxes, 3]
  points_rot_all = tf.einsum('bnij,hwbnj->hwbnj', rot, points_centered)

  # broadcast translation of shape (boxes, 3)
  points_t_all = points_rot_all + trans + pivots

  # compute difference between points and transformed points to obtain increments
  # which we can apply to the original points
  # only transform points where mask is 1 and where the object is moving
  diffs = points_t_all - points
  points_t = points + ((moving * masks) * diffs)

  return points_t


def _apply_camera_motion(points, motions):
  """Transform all points with global camera motion.
  Args:
    points: tensor of shape [mask_height, mask_width, num_boxes, batch_size, 3]
    motions: tensor of shape [batch_size, 9]
  returns:
    points_t: tensor of same shape as 'points'
  """
  motions = postprocess_motions(motions, has_pivot=False, has_moving=False)
  rot = tf.reshape(motions[:, 0:9], [-1, 3, 3])
  trans = motions[:, 9:12]

  # rotate the points with the camera rotation matrices
  # [batch_size, 3, 3], [h, w, num_boxes, batch_size, 3] ->
  # [h, w, num_boxes, batch_size, 3]
  points_rot = tf.einsum('bij,hwnbj->hwnbj', rotations, points)

  # broadcast translation
  points_t = points_rot + trans

  return points_t
