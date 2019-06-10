import tensorflow as tf

from . import ops as ops
from .config import cfg


def embedding_op(qIndices, embInit):
    with tf.variable_scope("qEmbeddings"):
        embInit = tf.to_float(embInit)
        embeddingsVar = tf.get_variable(
            "emb", initializer=embInit, dtype=tf.float32,
            trainable=(not cfg.WRD_EMB_FIXED))
        embeddings = tf.concat(
            [tf.zeros((1, cfg.WRD_EMB_DIM)), embeddingsVar], axis=0)
        questions = tf.nn.embedding_lookup(embeddings, qIndices)

    return questions


def encoder(qIndices, embInit, questionLengths, dropouts):
    questions = embedding_op(qIndices, embInit)
    with tf.variable_scope("encoder"):
        # rnns
        questionCntxWords, vecQuestions = ops.RNNLayer(
            questions, questionLengths, cfg.ENC_DIM, bi=True,
            cellType='LSTM', dropout=dropouts["encInput"], name="rnn0")

        # dropout for the question vector
        vecQuestions = tf.nn.dropout(vecQuestions, dropouts["question"])

    return questionCntxWords, vecQuestions
