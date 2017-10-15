# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------



def _pixels_to_3d(x, y, d, camera_intrinsics):
    f = camera_intrinsics['f']
    x0 = camera_intrinsics['x0']
    y0 = camera_intrinsics['y0']

    X = (x - x0) * d
    Y = (y - y0) * d
    Z = d

    return X, Y, Z

    def _3d_to_pixels(points, camera_intrinsics):
        """Project 3d coordinates to 2d pixel coordinates.
        """
        f = camera_intrinsics['f']
        x0 = camera_intrinsics['x0']
        y0 = camera_intrinsics['y0']

        X, Y, Z = tf.unstack(points, axis=3)

        x = f * X / Z + x0
        y = f * Y / Z + y0
        return tf.stack([x, y], axis=3)

def _apply_object_motions(point, motions, masks):
  """Transform points with per-object motions, weighted by per-pixel object masks.
  Args:
    points: tensor of shape (h, w, N, 3)
    motions: tensor of shape (N, 9)
    masks: tensor of shape (h, w, N)
  returns:
    points_t: tensor of same shape as 'points'
  """
  pivots = motions[:, 0:3]
  translations = motions[:, 3:6]
  sin_angles = tf.split(motions[:, 6:9], 3, axis=1)
  rotations = _compose_rotations(*sin_angles)

  # broadcast pivot point translation of shape (boxes, 3)
  points_centered = points - pivots

  # rotate the centered points with the rotation matrix of each mask,
  # (boxes, 3, 3), (h, w, boxes, 3) -> (h, w, boxes, 3)
  points_rot_all = tf.einsum('nij,hwnj->hwn', rotations, points_centered)

  # broadcast translation of shape (boxes, 3)
  points_t_all = points_rot_all + translations + pivots

  # compute difference between points and transformed points to obtain increments
  # which we can apply to the original points, weighted by the mask
  diffs = points_t_all - points
  points_t = points + (masks * diffs)

  return points_t

  def _apply_camera_motion(points, motion):
      """Transform all points with global camera motion.
      Args:
        points: tensor of shape (h, w, N, 3)
        motion: tensor of shape (9)
      returns:
        points_t: tensor of same shape as 'points'
      """
      pivot = motion[0:3]
      translation = motion[3:6]
      sin_angles = tf.split(tf.expand_dims(motion[6:9], axis=0), 3, axis=1)
      rotation = _compose_rotations(*sin_angles)[0, :]

      # broadcast pivot point translation of shape (3)
      points_centered = points - pivot

      # rotate the centered points with the camera rotation matrix
      # (3, 3), (h, w, boxes, 3) -> (h, w, boxes, 3)
      points_rot = tf.einsum('ij,hwnj->hwn', rotations, points_centered)

      # broadcast translation of shape (3)
      points_t = points_rot + translation + pivot

      return points_t

def motion_to_dense_flow(depth, motions, masks, camera_motion):
  """
  Args:
    depth: array with shape [height, width, 1].
    motions: array with shape [num_detections, 15].
    masks: array with shape [num_detections, height, width]
    camera_motion: array with shape [].

  Returns:
    Array with shape [height, width, 2].
  """
  h, w = depth.shape[:2]
  x_range = np.linspace(0, w - 1, w)
  y_range = np.linspace(0, h - 1, h)

  x, y = np.meshgrid(x_range, y_range)
  X, Y, Z = _pixels_to_3d(x, y, depth, TODO)
  P = np.stack([X, Y, Z], axis=2)

  for n in range(motions.shape[0]):
    R = np.reshape(motions[n, :9])
    t = np.reshape(motions[n, 9:12])
    p = np.reshape(motions[n, 12:])
    m = masks[i, :, :]

    P = P + m * ()
