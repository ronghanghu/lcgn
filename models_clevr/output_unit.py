import torch
from torch import nn

from . import ops as ops
from .config import cfg


class Classifier(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.outQuestion = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        in_dim = 3 * cfg.CTX_DIM if cfg.OUT_QUESTION_MUL else 2 * cfg.CTX_DIM
        self.classifier_layer = nn.Sequential(
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(in_dim, cfg.OUT_CLASSIFIER_DIM),
            nn.ELU(),
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(cfg.OUT_CLASSIFIER_DIM, num_choices))

    def forward(self, x_att, vecQuestions):
        eQ = self.outQuestion(vecQuestions)
        if cfg.OUT_QUESTION_MUL:
            features = torch.cat([x_att, eQ, x_att*eQ], dim=-1)
        else:
            features = torch.cat([x_att, eQ], dim=-1)
        logits = self.classifier_layer(features)
        return logits


class BboxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_offset_fcn = ops.Linear(cfg.CTX_DIM, 4)

    def forward(self, x_out, loc_scores):
        bbox_offset_fcn = self.bbox_offset_fcn(x_out)
        bbox_offset_flat = bbox_offset_fcn.view(-1, 4)

        assert len(x_out.size()) == 3
        batch_size = x_out.size(0)
        max_entity_num = x_out.size(1)
        slice_inds = (
            torch.arange(batch_size, device=x_out.device) * max_entity_num +
            torch.argmax(loc_scores, dim=-1))
        bbox_offset = bbox_offset_flat[slice_inds]

        return bbox_offset, bbox_offset_fcn
