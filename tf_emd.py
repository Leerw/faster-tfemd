import tensorflow as tf
from tensorflow.python.framework import ops
import os
import os.path as osp
import open3d as o3d
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

base_dir = osp.dirname(osp.abspath(__file__))

emd_module = tf.load_op_library(osp.join(base_dir, "tf_emd_so.so"))

def emd_match(xyz1, xyz2):
    """
    input:
        xyz1: (b, n, 3)
        xyz2: (b, m, 3), m == n
    return:
        assignment: (b, n), assignment from xyz1 to xyz2
    """
    return emd_module.emd_match(xyz1, xyz2)

def emd_cost(xyz1, xyz2, assignment):
    return emd_module.emd_cost(xyz1, xyz2, assignment)

@ops.RegisterShape("EmdMatch")
def _emd_match_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(3)
    b = shape1.dims[0]
    n = shape1.dims[1]
    return [tf.TensorShape([b, n])]

@ops.RegisterShape("EmdCost")
def _emd_cost_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(3)
    b = shape1.dims[0]
    n = shape1.dims[1]
    return [tf.TensorShape([b, n])]

@tf.RegisterGradient("EmdCost")
def _emd_cost_grad(op, grad_cost):
    """
    input:
        op
        grad_dist: (b, n)
    output:
        grad_xyz: (b, n, 3)
    """
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    assignment = op.inputs[2]
    grad_xyz = emd_module.emd_cost_grad(xyz1, xyz2, assignment)
    grad_xyz = grad_xyz * tf.expand_dims(grad_cost, 2)
    return [grad_xyz, None, None]

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points).astype(np.float32)
    return points


if __name__ == "__main__":
    pt_gt = read_pcd("4c880eae29fd97c1f9575f483c69ee5.pcd")
    pt_input = read_pcd("12d15ac778df6e4562b600da24e0965.pcd")

    pt_gt = tf.Variable(tf.convert_to_tensor(pt_gt))
    pt_input = tf.Variable(tf.convert_to_tensor(pt_input))

    pt_gt = tf.expand_dims(pt_gt, 0)
    pt_input = tf.expand_dims(pt_input, 0)

    print(pt_input, pt_gt)

    assignment = emd_match(pt_input, pt_gt)
    dist = emd_cost(pt_input, pt_gt, assignment)
    loss = tf.reduce_mean(tf.sqrt(dist))
    optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_ass, train_dist, train_loss, _ = sess.run([assignment, dist, loss, optimizer])
        print(train_ass)
        print(train_dist)
        print(train_loss)
