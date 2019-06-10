import tensorflow as tf

from . import ops as ops
from .config import cfg


def classifier(x_att, vecQuestions, dropouts, num_choices):
    with tf.variable_scope("outputUnit"):
        eQ = ops.linear(
            vecQuestions, cfg.CMD_DIM, cfg.CTX_DIM, name="outQuestion")
        features, dim = ops.concat(
            x_att, eQ, cfg.CTX_DIM, mul=cfg.OUT_QUESTION_MUL)

    with tf.variable_scope("classifier"):
        dims = [dim, cfg.OUT_CLASSIFIER_DIM, num_choices]
        logits = ops.FCLayer(features, dims, dropout=dropouts["output"])

    return logits


def bbox_regression(x_out, loc_scores):
    with tf.variable_scope('bbox_regression'):
        bbox_offset_fcn = ops.linear(
            x_out, cfg.CTX_DIM, 4, name='bbox_offset_fcn')
        bbox_offset_flat = tf.reshape(bbox_offset_fcn, [-1, 4])

        assert len(x_out.get_shape()) == 3
        batch_size = tf.shape(x_out)[0]
        max_entity_num = tf.shape(x_out)[1]
        slice_inds = tf.range(batch_size) * max_entity_num + tf.argmax(
            loc_scores, axis=-1, output_type=tf.int32)
        bbox_offset = tf.gather(bbox_offset_flat, slice_inds)

    return bbox_offset, bbox_offset_fcn
