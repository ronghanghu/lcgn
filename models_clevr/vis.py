import matplotlib; matplotlib.use('Agg')  # NoQA

import os
import json
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import functools

from .config import cfg
from util.boxes import bbox_iou


@functools.lru_cache(maxsize=32)
def _gen_connection_mesh(H, W):
    Hs = np.tile(np.arange(.5/H, 1, 1./H)[:, np.newaxis], (1, W))
    Ws = np.tile(np.arange(.5/W, 1, 1./W)[np.newaxis, :], (H, 1))
    mesh = [(x, y) for x, y in zip(Ws.reshape(-1), Hs.reshape(-1))]
    return mesh


def _find_txt_segs(words, keep):
    segs = []
    current_seg = []
    for n, k in enumerate(keep):
        if k:
            current_seg.append(words[n])
        else:
            if len(current_seg) > 0:
                segs.append('"%s"' % ' '.join(current_seg))
            current_seg = []
    if len(current_seg) > 0:
        segs.append('"%s"' % ' '.join(current_seg))
    return segs


def _extract_txt_att(words, atts, thresh=0.33):
    """
    Take at most 3 words that have at least thresh of the max attention.
    """
    atts_sorted = np.sort(atts)[::-1]
    att_min = max(atts_sorted[2], atts_sorted[0]*thresh)
    # collect those words above att_min
    keep = (atts >= att_min)
    vis_txt = ', '.join(_find_txt_segs(words, keep))
    return vis_txt


def vis_one_stepwise(img_path, words, txt_att, msg_att, final_obj_att,
                     save_path, vis_type, vqa_prediction=None, label=None,
                     answers=None, bbox_pred=None, bbox_gt=None):
    T = cfg.NET_LENGTH
    img = skimage.io.imread(img_path)

    h = plt.figure(figsize=((T+2)*5, 5))
    # Image and question
    plt.subplot(1, T+2, 1)
    plt.imshow(img)
    plt.axis('off')

    is_used = np.zeros((T, cfg.MODEL.H_FEAT*cfg.MODEL.W_FEAT), np.bool)
    if cfg.TEST.VIS_FILTER_EDGE:
        msg_att = msg_att.copy()
        # A node is used at step t if
        #   1) it is used in step t+1:T, or
        #   2) it sends a message to any node uses in t+1
        final_att_flat = final_obj_att.reshape(-1)
        is_final_obj = np.logical_and(
            final_att_flat >= cfg.TEST.VIS_FINAL_ABS_TH,
            final_att_flat/np.max(final_att_flat) >= cfg.TEST.VIS_FINAL_REL_TH)
        is_used[T-1] = is_final_obj
        for t in range(T-2, -1, -1):
            msg_att[t][~is_used[t+1], :] = 0
            is_used[t] = np.logical_or(
                is_used[t+1],
                np.max(msg_att[t], axis=0) >= cfg.TEST.VIS_MSG_TH)
    else:
        is_used[:] = True

    for t in range(T):
        att_txt = _extract_txt_att(words, txt_att[t, :len(words)])

        # output attention
        plt.subplot(1, T+2, t+2)
        plt.imshow(img)
        draw_grid_connection(
            msg_att[t], img.shape[0], img.shape[1], is_used[t])
        plt.xticks([], [])
        plt.yticks([], [])
        if t == (T-1) // 2:
            if vis_type == 'vqa':
                plt.title(' '.join(words) + '  pred: %s  gt: %s' % (
                    answers[vqa_prediction],
                    answers[label] if label >= 0 else '<n/a>'))
            elif vis_type == 'ref':
                plt.title(' '.join(words) +
                          '  pred: (red box)  gt: (yellow box)  IoU: %.3f' %
                          bbox_iou(bbox_pred, bbox_gt))
        plt.xlabel('t = %d - %s(%s)' % (t, 'text attention', att_txt))

    # the final object attention
    plt.subplot(1, T+2, T+2)
    plt.xticks([], [])
    plt.yticks([], [])
    if vis_type == 'vqa':
        img_with_att = attention_grid_interpolation(img, final_obj_att)
        plt.imshow(img_with_att)
        plt.xlabel('final visual attention')
    elif vis_type == 'ref':
        # img_with_att = attention_grid_interpolation(img, final_obj_att)
        # plt.imshow(img_with_att)
        plt.imshow(img)
        draw_final_grid_attention(final_obj_att, img.shape[0], img.shape[1])
        _print_bbox_xywh(bbox_gt, 'y')
        _print_bbox_xywh(bbox_pred, 'r')
        plt.xlabel('final localization')

    plt.savefig(save_path, bbox_inches='tight')
    with open(save_path.replace('.png', '') + '.json', 'w') as f:
        if vis_type == 'vqa':
            ans_pred, ans_gt = answers[vqa_prediction], answers[label]
            json.dump({'question': ' '.join(words), 'ans_pred': ans_pred,
                       'ans_gt': ans_gt}, f)
        elif vis_type == 'ref':
            json.dump({'refexp': ' '.join(words),
                       'bbox_pred': [float(b) for b in bbox_pred],
                       'bbox_gt': [float(b) for b in bbox_gt]}, f)
        else:
            raise ValueError('Unknow vis_type ' + str(vis_type))
    print('visualization saved to ' + save_path)
    plt.close(h)


def vis_batch_vqa(data_reader, batch, batch_res, start_idx, vis_dir):
    answers = data_reader.batch_loader.answer_dict.word_list
    num = min(len(batch['imageid_list']), cfg.TEST.NUM_VIS - start_idx)
    inds = range(num)
    for n in inds:
        imageId = batch['imageid_list'][n]
        split, imid = imageId.split('_')
        img_path = os.path.join(
            cfg.IMAGE_DIR, '%s/CLEVR_%s_%06d.png' % (split, split, int(imid)))
        save_name = '%08d_%s.png' % (
            start_idx, os.path.basename(img_path).split('.')[0])
        start_idx += 1
        save_path = os.path.join(vis_dir, save_name)
        words = [
            data_reader.batch_loader.vocab_dict.idx2word(n_w) for n_w in
            batch['input_seq_batch'][n, :batch['seq_length_batch'][n]]]
        label = batch['answer_label_batch'][n]
        vqa_prediction = batch_res['predictions'][n]
        txt_att = batch_res['vis']['txt_att'][n]
        msg_att = batch_res['vis']['msg_att'][n]
        final_obj_att = batch_res['vis']['final_obj_att'][n]
        vis_one_stepwise(img_path, words, txt_att, msg_att,
                         final_obj_att, save_path, vis_type='vqa',
                         vqa_prediction=vqa_prediction, label=label,
                         answers=answers)


def vis_batch_ref(data_reader, batch, batch_res, start_idx, vis_dir):
    num = min(len(batch['imageid_list']), cfg.TEST.NUM_VIS - start_idx)
    inds = range(num)
    for n in inds:
        imageId = batch['imageid_list'][n]
        split, imid = imageId.split('_')
        img_path = os.path.join(
            cfg.IMAGE_DIR, '%s/CLEVR_%s_%06d.png' % (split, split, int(imid)))
        save_name = '%08d_%s.png' % (
            start_idx, os.path.basename(img_path).split('.')[0])
        start_idx += 1
        save_path = os.path.join(vis_dir, save_name)
        words = [
            data_reader.batch_loader.vocab_dict.idx2word(n_w) for n_w in
            batch['input_seq_batch'][n, :batch['seq_length_batch'][n]]]
        bbox_pred = batch_res['bbox_predictions'][n]
        bbox_gt = batch['bbox_batch'][n]
        txt_att = batch_res['vis']['txt_att'][n]
        msg_att = batch_res['vis']['msg_att'][n]
        final_obj_att = batch_res['vis']['grounder_softmax'][n]
        vis_one_stepwise(img_path, words, txt_att, msg_att,
                         final_obj_att, save_path, vis_type='ref',
                         bbox_pred=bbox_pred, bbox_gt=bbox_gt)


def _print_normalized_bboxes(bboxes, img_h, img_w, color='r', print_ctr=False):
    for bbox in bboxes:
        _print_normalized_bbox(
            bbox, img_h, img_w, color=color, print_ctr=print_ctr)


def _print_normalized_bbox(bbox, img_h, img_w, color='r', print_ctr=False):
    x1, y1, x2, y2 = bbox
    x1 = int(round(x1*img_w))
    y1 = int(round(y1*img_h))
    x2 = min(int(round(x2*img_w)), img_w-1)
    y2 = min(int(round(y2*img_h)), img_h-1)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color)
    if print_ctr:
        plt.plot([(x1 + x2) / 2], [(y1 + y2) / 2], '*', color=color,
                 markersize=12)


def _print_bbox_xywh(bbox, color='r'):
    x1, y1, h, w = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color, linewidth=2.)


def draw_bbox_connection(bboxes, att, img_h, img_w, show_vis, color='r'):
    is_valid = np.any(bboxes > 0., axis=1)
    for n in range(len(bboxes)):
        if not (is_valid[n] and show_vis[n]):
            break
        x_n = (bboxes[n, 0] + bboxes[n, 2]) / 2 * img_w
        y_n = (bboxes[n, 1] + bboxes[n, 3]) / 2 * img_h
        for m in range(len(bboxes)):
            if not is_valid[m]:
                break
            x_m = (bboxes[m, 0] + bboxes[m, 2]) / 2 * img_w
            y_m = (bboxes[m, 1] + bboxes[m, 3]) / 2 * img_h
            w = att[n, m]
            if w >= cfg.TEST.VIS_MSG_TH:
                size = w * cfg.TEST.VIS_EDGE_SCALE
                plt.plot([x_n, x_m], [y_n, y_m], color=color, linewidth=3*size)
                plt.plot([x_n], [y_n], '+', color='r', markersize=15*size)
                plt.plot([x_m], [y_m], '*', color='b', markersize=15*size)


def draw_grid_connection(att, img_h, img_w, show_vis, color='r'):
    M, M = att.shape
    mesh = _gen_connection_mesh(cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT)
    for ms in range(M):
        if not show_vis[ms]:
            continue
        for mt in range(M):
            w = att[ms, mt]
            if w >= cfg.TEST.VIS_MSG_TH:
                xs, ys = mesh[ms]
                xt, yt = mesh[mt]
                xs = xs * img_w
                xt = xt * img_w
                ys = ys * img_h
                yt = yt * img_h
                size = w * cfg.TEST.VIS_EDGE_SCALE
                plt.plot([xs, xt], [ys, yt], color=color, linewidth=3*size)
                plt.plot([xs], [ys], '+', color='r', markersize=15*size)
                plt.plot([xt], [yt], '*', color='b', markersize=15*size)


def attention_bbox_interpolation(im, bboxes, att):
    softmax = att
    softmax = np.squeeze(softmax)
    assert len(softmax) == len(bboxes)

    img_h, img_w = im.shape[:2]
    opacity = np.zeros((img_h, img_w), np.float32)
    for bbox, weight in zip(bboxes, softmax):
        x1, y1, x2, y2 = bbox
        x1 = int(round(x1*img_w))
        y1 = int(round(y1*img_h))
        x2 = min(int(round(x2*img_w)), img_w-1)
        y2 = min(int(round(y2*img_h)), img_h-1)
        opacity[y1:y2, x1:x2] += weight
    assert np.all(opacity <= 1.)

    opacity = opacity*0.67 + 0.33
    vis_im = im * opacity[..., np.newaxis]
    vis_im = vis_im.astype(im.dtype)
    return vis_im


def attention_grid_interpolation(im, att):
    softmax = np.reshape(att, (cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT))
    att_reshaped = skimage.transform.resize(softmax, im.shape[:2], order=3)
    # normalize the attention
    # make sure the 255 alpha channel is at least 3x uniform attention
    att_reshaped /= np.maximum(np.max(att_reshaped), 3. / att.size)
    att_reshaped = att_reshaped[..., np.newaxis]

    # make the attention area brighter than the rest of the area
    vis_im = att_reshaped * im + (1-att_reshaped) * im * .33
    vis_im = vis_im.astype(im.dtype)
    return vis_im


def draw_final_grid_attention(final_obj_att, img_h, img_w, color='r'):
    mesh = _gen_connection_mesh(cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT)
    max_idx = np.argmax(final_obj_att)
    xs, ys = mesh[max_idx]
    xs = xs * img_w
    ys = ys * img_h
    plt.plot([xs], [ys], '*', color='r', markersize=15)
