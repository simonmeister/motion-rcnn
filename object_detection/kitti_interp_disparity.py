import tensorflow as tf
import os
import subprocess

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory for datasets.')
flags.DEFINE_string('interp_path', '../devkit/cpp/interp_disp', 
                    'Path for interp_disp executable.')
FLAGS = flags.FLAGS


def main(_):
  dataset_dir = os.path.join(FLAGS.data_dir, 'data_scene_flow', 'training')
  src_dir_0 = os.path.join(dataset_dir, 'disp_occ_0')
  src_dir_1 = os.path.join(dataset_dir, 'disp_occ_1')
  target_dir_0 = os.path.join(dataset_dir, 'disp_occ_0_interp')
  target_dir_1 = os.path.join(dataset_dir, 'disp_occ_1_interp')
  #os.makedirs(target_dir_0)
  #os.makedirs(target_dir_1)
  for i in range(200):
    n = str(i).zfill(6)
    src_0 = os.path.join(src_dir_0, n + '_10.png')
    src_1 = os.path.join(src_dir_1, n + '_10.png')
    target_0 = os.path.join(target_dir_0, n + '_10.png')
    target_1 = os.path.join(target_dir_1, n + '_10.png')
    subprocess.call("{} {} {}".format(FLAGS.interp_path, src_0, target_0), shell=True)
    subprocess.call("{} {} {}".format(FLAGS.interp_path, src_1, target_1), shell=True)


if __name__ == '__main__':
  tf.app.run()
