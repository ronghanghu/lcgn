import tensorflow as tf
import numpy as np

from . import ops as ops
from .config import cfg


class LCGN():
    def __init__(self, images, q_encoding, lstm_outputs, batch_size,
                 q_length, entity_num, dropouts, train, reuse=None):
        self.q_encoding = q_encoding
        self.lstm_outputs = lstm_outputs
        self.images = images
        self.batch_size = batch_size
        self.q_length = q_length
        self.entity_num = entity_num
        self.max_entity_num = tf.shape(images)[-2]

        self.dropouts = dropouts
        self.train = train
        self.reuse = reuse
        self.attentions = {"edge": [], "question": []}

        # Apply LCGN on local features
        # 1. initialize x_loc and x_ctx
        self.loc_ctx_init()
        # 2. iterative message passing
        x_ctx_new = self.x_ctx_init
        for i in range(cfg.MSG_ITER_NUM):
            self.iteration = i
            x_ctx_new = self.run_message_passing_iter(x_ctx_new)
        # 3. merge x_loc and x_ctx for output
        self.x_out = ops.linear(
            tf.concat([self.x_loc, x_ctx_new], axis=-1),
            cfg.CTX_DIM*2, cfg.CTX_DIM, name="combined_kb")

    def extract_textual_command(self):
        # Textual command extraction
        q_cmd = ops.linear(
            self.q_encoding, cfg.CMD_DIM, cfg.CMD_DIM, name="qInput",
            reuse=(self.iteration > 0))
        q_cmd = ops.activations[cfg.CMD_INPUT_ACT](q_cmd)
        q_cmd = ops.linear(
            q_cmd, cfg.CMD_DIM, cfg.CMD_DIM, name="qInput%d" % self.iteration,
            reuse=None)  # This linear layer is not shared across iterations
        with tf.variable_scope("control", reuse=(self.iteration > 0)):
            # compute attention distribution over words and summarize them
            interactions = tf.expand_dims(q_cmd, axis=1) * self.lstm_outputs
            logits = ops.inter2logits(interactions, cfg.CMD_DIM)
            attention = tf.nn.softmax(ops.expMask(logits, self.q_length))
            self.attentions["question"].append(attention)
            cmd = ops.att2Smry(attention, self.lstm_outputs)

        return cmd

    def propagate_message(self, x_ctx, cmd):
        with tf.variable_scope("read", reuse=(self.iteration > 0)):
            # Language Conditioned message passing
            # Step 1: join x_loc and x_ctx
            dim = cfg.CTX_DIM
            x_ctx = ops.applyVarDpMask(
                x_ctx, self.ctx_drop_mask, self.dropouts["memory"])
            proj = {
                "dim": dim, "shared": False, "dropout": self.dropouts["read"]}
            x_mul, _ = ops.mul(
                self.x_loc, x_ctx, dim=dim, proj=proj, name="memInter",
                expandY=False)
            x_joint = tf.concat([self.x_loc, x_ctx, x_mul], axis=-1)

            # Step 2: compute edge weights
            queries = ops.linear(x_joint, dim*3, dim, name="queries")
            keys = ops.linear(x_joint, dim*3, dim, name="keys")
            vals = ops.linear(x_joint, dim*3, dim, name="vals")
            p_keys = ops.linear(cmd, cfg.CMD_DIM, dim, name="proj_keys")
            p_vals = ops.linear(cmd, cfg.CMD_DIM, dim, name="proj_vals")
            keys, _ = ops.mul(keys, p_keys, dim, name="ctrl_keys")
            vals, _ = ops.mul(vals, p_vals, dim, name="ctrl_vals")

            # edge_prob - (batch_size, max_entity_num, max_entity_num) shape
            edge_score = (
                tf.matmul(queries, keys, transpose_b=True) / np.sqrt(dim))
            edge_score = apply_mask2d(edge_score, self.entity_num)
            edge_prob = tf.nn.softmax(edge_score, axis=-1)
            self.attentions['edge'].append(edge_prob)

            # Step 3: propagate message
            message = tf.matmul(edge_prob, vals)
            x_ctx_new = tf.concat([x_ctx, message], axis=-1)
            x_ctx_new = ops.linear(x_ctx_new, dim*2, dim, name="mem_update")

        return x_ctx_new

    def run_message_passing_iter(self, x_ctx, scope=None):
        with tf.variable_scope('LCGNCell', reuse=self.reuse):
            # textual command extraction
            cmd = self.extract_textual_command()
            # language-conditioned message passing
            x_ctx_new = self.propagate_message(x_ctx, cmd)

        return x_ctx_new

    def loc_ctx_init(self):
        if cfg.STEM_NORMALIZE:
            self.images = tf.nn.l2_normalize(self.images, axis=-1)
        assert cfg.STEM_LINEAR != cfg.STEM_CNN
        if cfg.STEM_LINEAR:
            self.x_loc = ops.linear(
                self.images, cfg.D_FEAT, cfg.CTX_DIM, name="initKB")
            self.x_loc = tf.nn.dropout(self.x_loc, self.dropouts["stem"])
        elif cfg.STEM_CNN:
            dims = [cfg.D_FEAT, cfg.STEM_CNN_DIM, cfg.CTX_DIM]
            self.x_loc = tf.reshape(
                self.images, [-1, cfg.H_FEAT, cfg.W_FEAT, cfg.D_FEAT])
            self.x_loc = ops.CNNLayer(
                self.x_loc, dims, dropout=self.dropouts["stem"],
                kernelSizes=[3, 3])
        self.x_loc = tf.reshape(self.x_loc, (self.batch_size, -1, cfg.CTX_DIM))
        if cfg.STEM_RENORMALIZE:
            self.x_loc = tf.nn.l2_normalize(self.x_loc, axis=-1)

        prm = tf.get_variable(
            "initMem", shape=(1, 1, cfg.CTX_DIM),
            initializer=tf.random_normal_initializer())
        self.x_ctx_init = tf.tile(
            prm, [self.batch_size, self.max_entity_num, 1])

        # initialize x_ctx variational dropout mask
        self.ctx_drop_mask = ops.generateVarDpMask(
            (self.batch_size, self.max_entity_num, cfg.CTX_DIM),
            self.dropouts["memory"])


def apply_mask2d(seq, seqLength):
    maxLength = tf.shape(seq)[-1]
    seq_mask_1d = tf.sequence_mask(seqLength, maxLength)
    seq_mask_2d = tf.logical_and(
        seq_mask_1d[:, :, None], seq_mask_1d[:, None, :])
    mask = tf.to_float(tf.logical_not(seq_mask_2d)) * (-1e30)
    masked = seq + mask
    return masked
