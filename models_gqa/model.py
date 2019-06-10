from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg
from .lcgn import LCGN
from .input_unit import Encoder
from .output_unit import Classifier


class SingleHop(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)
        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, vecQuestions, imagesObjectNum):
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, imagesObjectNum)
        att = F.softmax(raw_att, dim=-1)

        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att


class LCGNnet(nn.Module):
    def __init__(self, num_vocab, num_choices):
        super().__init__()
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.randn(num_vocab-1, cfg.WRD_EMB_DIM)
        self.num_vocab = num_vocab
        self.num_choices = num_choices
        self.encoder = Encoder(embeddingsInit)
        self.lcgn = LCGN()
        self.single_hop = SingleHop()
        self.classifier = Classifier(num_choices)

    def forward(self, batch):
        batchSize = len(batch['image_feat_batch'])
        questionIndices = torch.from_numpy(
            batch['input_seq_batch'].astype(np.int64)).cuda()
        questionLengths = torch.from_numpy(
            batch['seq_length_batch'].astype(np.int64)).cuda()
        answerIndices = torch.from_numpy(
            batch['answer_label_batch'].astype(np.int64)).cuda()
        images = torch.from_numpy(
            batch['image_feat_batch'].astype(np.float32)).cuda()
        imagesObjectNum = torch.from_numpy(
            np.sum(batch['image_valid_batch'].astype(np.int64), axis=1)).cuda()

        # LSTM
        questionCntxWords, vecQuestions = self.encoder(
            questionIndices, questionLengths)

        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum)

        # Single-Hop
        x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
        logits = self.classifier(x_att, vecQuestions)

        predictions, num_correct = self.add_pred_op(logits, answerIndices)
        loss = self.add_answer_loss_op(logits, answerIndices)

        return {"predictions": predictions,
                "batch_size": int(batchSize),
                "num_correct": int(num_correct),
                "loss": loss,
                "accuracy": float(num_correct * 1. / batchSize)}

    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctNum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()

        return preds, correctNum

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            loss = F.cross_entropy(logits, answers)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerDist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss


class LCGNwrapper():
    def __init__(self, num_vocab, num_choices):
        self.model = LCGNnet(num_vocab, num_choices).cuda()
        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            self.trainable_params, lr=cfg.TRAIN.SOLVER.LR)
        self.lr = cfg.TRAIN.SOLVER.LR

        if cfg.USE_EMA:
            self.ema_param_dict = {
                name: p for name, p in self.model.named_parameters()
                if p.requires_grad}
            self.ema = ops.ExponentialMovingAverage(
                self.ema_param_dict, decay=cfg.EMA_DECAY_RATE)
            self.using_ema_params = False

    def train(self, training=True):
        self.model.train(training)
        if training:
            self.set_params_from_original()
        else:
            self.set_params_from_ema()

    def eval(self):
        self.train(False)

    def state_dict(self):
        # Generate state dict in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict() if cfg.USE_EMA else None
        }

        # restore original mode
        self.train(current_mode)

    def load_state_dict(self, state_dict):
        # Load parameters in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            print('Optimizer does not exist in checkpoint! '
                  'Loaded only model parameters.')

        if cfg.USE_EMA:
            if 'ema' in state_dict and state_dict['ema'] is not None:
                self.ema.load_state_dict(state_dict['ema'])
            else:
                print('cfg.USE_EMA is True, but EMA does not exist in '
                      'checkpoint! Using model params to initialize EMA.')
                self.ema.load_state_dict(
                    {k: p.data for k, p in self.ema_param_dict.items()})

        # restore original mode
        self.train(current_mode)

    def set_params_from_ema(self):
        if (not cfg.USE_EMA) or self.using_ema_params:
            return

        self.original_state_dict = deepcopy(self.model.state_dict())
        self.ema.set_params_from_ema(self.ema_param_dict)
        self.using_ema_params = True

    def set_params_from_original(self):
        if (not cfg.USE_EMA) or (not self.using_ema_params):
            return

        self.model.load_state_dict(self.original_state_dict)
        self.using_ema_params = False

    def run_batch(self, batch, train, lr=None):
        assert train == self.model.training
        assert (not train) or (lr is not None), 'lr must be set for training'

        if train:
            if lr != self.lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.optimizer.zero_grad()
            batch_res = self.model.forward(batch)
            loss = batch_res['loss']
            loss.backward()
            if cfg.TRAIN.CLIP_GRADIENTS:
                nn.utils.clip_grad_norm_(
                    self.trainable_params, cfg.TRAIN.GRAD_MAX_NORM)
            self.optimizer.step()
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            with torch.no_grad():
                batch_res = self.model.forward(batch)

        return batch_res
