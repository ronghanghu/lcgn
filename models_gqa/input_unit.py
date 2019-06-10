import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .config import cfg


class Encoder(nn.Module):
    def __init__(self, embInit):
        super().__init__()
        self.embeddingsVar = nn.Parameter(
            torch.Tensor(embInit), requires_grad=(not cfg.WRD_EMB_FIXED))
        self.enc_input_drop = nn.Dropout(1 - cfg.encInputDropout)
        self.rnn0 = BiLSTM()
        self.question_drop = nn.Dropout(1 - cfg.qDropout)

    def forward(self, qIndices, questionLengths):
        # Word embedding
        embeddingsVar = self.embeddingsVar.cuda()
        embeddings = torch.cat(
            [torch.zeros(1, cfg.WRD_EMB_DIM, device='cuda'), embeddingsVar],
            dim=0)
        questions = F.embedding(qIndices, embeddings)
        questions = self.enc_input_drop(questions)

        # RNN (LSTM)
        questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)
        vecQuestions = self.question_drop(vecQuestions)

        return questionCntxWords, vecQuestions


class BiLSTM(nn.Module):
    def __init__(self, forget_gate_bias=1.):
        super().__init__()
        self.bilstm = torch.nn.LSTM(
            input_size=cfg.WRD_EMB_DIM, hidden_size=cfg.ENC_DIM // 2,
            num_layers=1, batch_first=True, bidirectional=True)

        d = cfg.ENC_DIM // 2

        # initialize LSTM weights (to be consistent with TensorFlow)
        fan_avg = (d*4 + (d+cfg.WRD_EMB_DIM)) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

        # initialize LSTM forget gate bias (to be consistent with TensorFlow)
        self.bilstm.bias_ih_l0.data[...] = 0.
        self.bilstm.bias_ih_l0.data[d:2*d] = forget_gate_bias
        self.bilstm.bias_hh_l0.data[...] = 0.
        self.bilstm.bias_hh_l0.requires_grad = False
        self.bilstm.bias_ih_l0_reverse.data[...] = 0.
        self.bilstm.bias_ih_l0_reverse.data[d:2*d] = forget_gate_bias
        self.bilstm.bias_hh_l0_reverse.data[...] = 0.
        self.bilstm.bias_hh_l0_reverse.requires_grad = False

    def forward(self, questions, questionLengths):
        # sort samples according to question length (descending)
        sorted_lengths, indices = torch.sort(questionLengths, descending=True)
        sorted_questions = questions[indices]
        _, desorted_indices = torch.sort(indices, descending=False)

        # pack questions for LSTM forwarding
        packed_questions = nn.utils.rnn.pack_padded_sequence(
            sorted_questions, sorted_lengths, batch_first=True)
        packed_output, (sorted_h_n, _) = self.bilstm(packed_questions)
        sorted_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=questions.size(1))
        sorted_h_n = torch.transpose(sorted_h_n, 1, 0).reshape(
            questions.size(0), -1)

        # sort back to the original sample order
        output = sorted_output[desorted_indices]
        h_n = sorted_h_n[desorted_indices]

        return output, h_n
