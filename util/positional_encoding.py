import numpy as np


def get_positional_encoding(H, W, dim=128):
    """
    2D positional encoding, following
        https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    """

    assert dim % 4 == 0, 'dim must be a multiply of 4 (h/w x sin/cos)'
    c_period = 10000. ** np.linspace(0., 1., dim // 4)
    h_vec = np.tile(np.arange(0, H).reshape((H, 1, 1)), (1, W, 1)) / c_period
    w_vec = np.tile(np.arange(0, W).reshape((1, W, 1)), (H, 1, 1)) / c_period
    position_encoding = np.concatenate(
        (np.sin(h_vec), np.cos(h_vec), np.sin(w_vec), np.cos(w_vec)), axis=-1)
    position_encoding = position_encoding.reshape((1, H, W, dim))
    return position_encoding
