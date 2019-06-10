import tensorflow as tf
import numpy as np

from . import ops as ops
from .config import cfg
from .lcgn import LCGN
from .input_unit import encoder
from .output_unit import classifier


class LCGNnet():
    def __init__(self, num_vocab, num_choices, gpusNum=1):
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.randn(num_vocab-1, cfg.WRD_EMB_DIM)
        self.num_vocab = num_vocab
        self.num_choices = num_choices
        self.build(embeddingsInit, gpusNum)

    def LCGNnetwork(self, images, vecQuestions, questionCntxWords,
                    questionLengths, reuse=None):
        with tf.variable_scope("LCGNnetwork", reuse=reuse):
            self.lcgn = LCGN(
                images=images,
                q_encoding=vecQuestions,
                lstm_outputs=questionCntxWords,
                batch_size=self.batchSize,
                q_length=questionLengths,
                entity_num=self.imagesObjectNum,
                dropouts=self.dropouts,
                train=self.train,
                reuse=reuse)

            self.x_out = self.lcgn.x_out

            # task-specific output: single-hop attention over x_out
            x_att = self.single_hop(self.x_out, vecQuestions, cfg.CTX_DIM)

        return x_att

    def single_hop(self, kb, vecQuestions, dim):
        with tf.variable_scope('single_hop'):
            proj_q = ops.linear(vecQuestions, cfg.ENC_DIM, dim, name="proj_q")
            interactions = tf.nn.l2_normalize(
                kb * proj_q[:, tf.newaxis, :], axis=-1)
            att = ops.inter2att(
                interactions, dim, mask=self.imagesObjectNum)
            self.single_hop_att = att
            x_att = tf.squeeze(tf.matmul(att[:, tf.newaxis, :], kb), axis=[1])

        return x_att

    def add_pred_op(self, logits, answers):
        with tf.variable_scope("pred"):
            if cfg.MASK_PADUNK_IN_LOGITS:
                mask = tf.to_float(
                    tf.sequence_mask([2], self.num_choices)) * (-1e30)
                logits += mask

            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            corrects = tf.to_float(tf.equal(preds, answers))
            correctNum = tf.reduce_sum(corrects)

        return preds, correctNum

    def add_vis_op(self):
        with tf.variable_scope('vis'):
            vis_dict = {}
            vis_dict['txt_att'] = tf.stack(
                self.lcgn.attentions['question'], axis=1)
            vis_dict['msg_att'] = tf.stack(
                self.lcgn.attentions['edge'], axis=1)
            vis_dict['final_obj_att'] = self.single_hop_att

        return vis_dict

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=answers, logits=logits)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerDist = tf.one_hot(answers, self.num_choices)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=answerDist, logits=logits)
            losses = tf.reduce_sum(losses, axis=-1)
        else:
            raise Exception("non-identified loss")
        loss = tf.reduce_mean(losses)
        return loss

    # Creates optimizer (adam)
    def add_optimizer_op(self):
        with tf.variable_scope("trainAddOptimizer"):
            self.globalStep = tf.Variable(
                0, dtype=tf.int32, trainable=False, name="globalStep")
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        return optimizer

    def add_training_op(self, optimizer, gradients_vars):
        with tf.variable_scope("train"):
            gradients, variables = zip(*gradients_vars)
            norm = tf.global_norm(gradients)

            # gradient clipping
            if cfg.TRAIN.CLIP_GRADIENTS:
                clippedGradients, _ = tf.clip_by_global_norm(
                    gradients, cfg.TRAIN.GRAD_MAX_NORM, use_norm=norm)
                gradients_vars = zip(clippedGradients, variables)

            # updates ops (for batch norm) and train op
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                train = optimizer.apply_gradients(
                    gradients_vars, global_step=self.globalStep)

            # exponential moving average
            if cfg.USE_EMA:
                ema = tf.train.ExponentialMovingAverage(
                    decay=cfg.EMA_DECAY_RATE)
                maintainAveragesOp = ema.apply(tf.trainable_variables())
                with tf.control_dependencies([train]):
                    trainAndUpdateOp = tf.group(maintainAveragesOp)
                train = trainAndUpdateOp
                self.emaDict = ema.variables_to_restore()

        return train, norm

    def build(self, embeddingsInit, gpusNum):
        self.add_placeholders()
        self.noOp = tf.no_op()
        self.optimizer = self.add_optimizer_op()

        self.gradientVarsList = []
        self.lossList = []
        self.answerLossList = []
        self.correctNumList = []
        self.answerAccList = []
        self.predsList = []
        self.visList = []

        with tf.variable_scope("lcgnModel"):
            for i in range(gpusNum):
                with tf.device("/gpu:%d" % i), tf.name_scope("tower%d" % i):
                    self.init_tower_batch(i, gpusNum, self.batchSizeAll)
                    self.build_tower(embeddingsInit)
                    tf.get_variable_scope().reuse_variables()
            self.average_across_towers(gpusNum)

        self.trainOp, self.gradNorm = self.add_training_op(
            self.optimizer, self.gradientVarsAll)

    def build_tower(self, embeddingsInit):
        tower_loss = tf.constant(0.)

        # embed questions words
        questionCntxWords, vecQuestions = encoder(
            self.questionIndices, embeddingsInit, self.questionLengths,
            self.dropouts)

        x_att = self.LCGNnetwork(
            self.images, vecQuestions, questionCntxWords, self.questionLengths)

        # visualization
        vis_dict = self.add_vis_op()
        self.visList.append(vis_dict)

        logits = classifier(
            x_att, vecQuestions, self.dropouts, self.num_choices)

        # compute loss, predictions, accuracy
        predictions, num_correct = self.add_pred_op(logits, self.answerIndices)
        self.predsList.append(predictions)
        self.correctNumList.append(num_correct)

        answerLoss = self.add_answer_loss_op(logits, self.answerIndices)
        self.answerLossList.append(answerLoss)
        tower_loss += answerLoss
        self.lossList.append(tower_loss)

        # compute gradients
        gradient_vars = self.optimizer.compute_gradients(tower_loss)
        self.gradientVarsList.append(gradient_vars)

    def init_tower_batch(self, towerI, towersNum, dataSize):
        towerBatchSize = tf.floordiv(dataSize, towersNum)
        start = towerI * towerBatchSize
        end = (towerI+1)*towerBatchSize if towerI < towersNum-1 else dataSize

        self.questionIndices = self.questionIndicesAll[start:end]
        self.questionLengths = self.questionLengthsAll[start:end]
        self.images = self.imagesAll[start:end]
        self.imagesObjectNum = self.imagesObjectNumAll[start:end]
        self.answerIndices = self.answerIndicesAll[start:end]
        self.batchSize = end - start

    def average_across_towers(self, gpusNum):
        if gpusNum == 1:
            self.lossAll = self.lossList[0]
            self.answerLossAll = self.answerLossList[0]
            self.correctNumAll = self.correctNumList[0]
            self.predsAll = self.predsList[0]
            self.gradientVarsAll = self.gradientVarsList[0]
            self.visAll = self.visList[0]
        else:
            self.lossAll = tf.reduce_mean(tf.stack(self.lossList))
            self.answerLossAll = tf.reduce_mean(tf.stack(self.answerLossList))
            self.correctNumAll = tf.reduce_sum(tf.stack(self.correctNumList))
            self.predsAll = tf.concat(self.predsList, axis=0)

            self.gradientVarsAll = []
            for grads_var in zip(*self.gradientVarsList):
                gradients, variables = zip(*grads_var)
                if gradients[0] is not None:
                    avgGradient = tf.reduce_mean(tf.stack(gradients), axis=0)
                else:
                    avgGradient = None
                var = variables[0]
                grad_var = (avgGradient, var)
                self.gradientVarsAll.append(grad_var)

            self.visAll = {}
            for key in self.visList[0]:
                self.visAll[key] = tf.concat(
                    [vis_dict[key] for vis_dict in self.visList], axis=0)

    def add_placeholders(self):
        with tf.variable_scope("Placeholders"):
            # questions
            self.questionIndicesAll = tf.placeholder(
                tf.int32, shape=(None, None))
            self.questionLengthsAll = tf.placeholder(tf.int32, shape=(None,))

            # images; put image known dimension as last dim?
            self.imagesAll = tf.placeholder(
                tf.float32, shape=(None, None, None))
            self.imagesObjectNumAll = tf.placeholder(tf.int32, shape=(None,))

            # answers
            self.answerIndicesAll = tf.placeholder(tf.int32, shape=(None, ))

            # optimization
            self.lr = tf.placeholder(tf.float32, shape=())
            self.train = tf.placeholder(tf.bool, shape=())
            self.batchSizeAll = tf.shape(self.questionIndicesAll)[0]

            # dropouts
            self.dropouts = {
                "encInput": tf.placeholder(tf.float32, shape=()),
                "stem": tf.placeholder(tf.float32, shape=()),
                "question": tf.placeholder(tf.float32, shape=()),
                "read": tf.placeholder(tf.float32, shape=()),
                "memory": tf.placeholder(tf.float32, shape=()),
                "output": tf.placeholder(tf.float32, shape=()),
            }

    def create_feed_dict(self, batch, train, lr):
        feedDict = {
            self.questionIndicesAll: batch['input_seq_batch'],
            self.questionLengthsAll: batch['seq_length_batch'],
            self.answerIndicesAll: batch['answer_label_batch'],
            self.imagesAll: batch['image_feat_batch'],
            self.imagesObjectNumAll: np.sum(batch['image_valid_batch'], axis=1),  # NoQA
            self.dropouts["encInput"]: cfg.encInputDropout if train else 1.,
            self.dropouts["stem"]: cfg.stemDropout if train else 1.,
            self.dropouts["question"]: cfg.qDropout if train else 1.,
            self.dropouts["memory"]: cfg.memoryDropout if train else 1.,
            self.dropouts["read"]: cfg.readDropout if train else 1.,
            self.dropouts["output"]: cfg.outputDropout if train else 1.,
            self.lr: lr if train else 0.,
            self.train: train
        }
        return feedDict

    def run_batch(self, sess, batch, train, vis=False, lr=None):
        assert (not train) or (lr is not None), 'lr must be set for training'

        batchSizeOp = self.batchSizeAll
        trainOp = self.trainOp if train else self.noOp
        gradNormOp = self.gradNorm if train else self.noOp
        visOp = self.visAll if vis else self.noOp

        feed_dict = self.create_feed_dict(batch, train, lr)
        batchSize, _, loss, predictions, num_correct, gradNorm, visRes = \
            sess.run([batchSizeOp, trainOp, self.lossAll, self.predsAll,
                     self.correctNumAll, gradNormOp, visOp],
                     feed_dict=feed_dict)

        return {"predictions": predictions,
                "batch_size": int(batchSize),
                "num_correct": int(num_correct),
                "loss": float(loss),
                "accuracy": float(num_correct * 1. / batchSize),
                "grad_norm": float(gradNorm) if train else -1.,
                "vis": visRes}
