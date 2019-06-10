import os
import tensorflow as tf
import json

from models_gqa.model import LCGNnet
from models_gqa.config import build_cfg_from_argparse
from models_gqa.vis import vis_batch_vqa
from util.gqa_train.data_reader import DataReader

# Load config
cfg = build_cfg_from_argparse()

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))


def load_train_data(max_num=0):
    imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
    scene_graph_file = cfg.SCENE_GRAPH_FILE % \
        cfg.TRAIN.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
    data_reader = DataReader(
        imdb_file, shuffle=True, max_num=max_num,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        feature_type=cfg.FEAT_TYPE,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
        objects_max_num=cfg.W_FEAT,
        scene_graph_file=scene_graph_file,
        vocab_name_file=cfg.VOCAB_NAME_FILE,
        vocab_attr_file=cfg.VOCAB_ATTR_FILE,
        add_pos_enc=cfg.ADD_POS_ENC,
        pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices


def run_train_on_data(model, data_reader_train, saver_train, run_eval=False,
                      data_reader_eval=None, saver_eval=None):
    lr = cfg.TRAIN.SOLVER.LR
    correct, total, loss_sum, batch_num = 0, 0, 0., 0
    for batch, n_sample, e in data_reader_train.batches(one_pass=False):
        n_epoch = cfg.TRAIN.START_EPOCH + e
        if n_sample == 0 and n_epoch > cfg.TRAIN.START_EPOCH:
            print('')
            # save snapshot
            snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, n_epoch)
            saver_train.save(sess, snapshot_file, write_meta_graph=False)
            # run evaluation
            if run_eval:
                saver_eval.restore(sess, snapshot_file)
                run_eval_on_data(model, data_reader_eval)
                saver_train.restore(sess, snapshot_file)
            # clear stats
            correct, total, loss_sum, batch_num = 0, 0, 0., 0
        if n_epoch >= cfg.TRAIN.MAX_EPOCH:
            break
        batch_res = model.run_batch(sess, batch, train=True, lr=lr)
        correct += batch_res['num_correct']
        total += batch_res['batch_size']
        loss_sum += batch_res['loss']
        batch_num += 1
        print('\rTrain E %d S %d: avgL=%.4f, avgA=%.4f, lr=%.1e' % (
                n_epoch+1, total, loss_sum/batch_num, correct/total, lr),
              end='')


def load_eval_data(max_num=0):
    imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
    scene_graph_file = cfg.SCENE_GRAPH_FILE % \
        cfg.TEST.SPLIT_VQA.replace('_balanced', '').replace('_all', '')
    data_reader = DataReader(
        imdb_file, shuffle=False, max_num=max_num,
        batch_size=cfg.TEST.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        feature_type=cfg.FEAT_TYPE,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        objects_feature_dir=cfg.OBJECTS_FEATURE_DIR,
        objects_max_num=cfg.W_FEAT,
        scene_graph_file=scene_graph_file,
        vocab_name_file=cfg.VOCAB_NAME_FILE,
        vocab_attr_file=cfg.VOCAB_ATTR_FILE,
        add_pos_enc=cfg.ADD_POS_ENC,
        pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    return data_reader, num_vocab, num_choices


def run_eval_on_data(model, data_reader_eval, pred=False, vis=False,
                     vis_dir=None):
    predictions = []
    answer_tokens = data_reader_eval.batch_loader.answer_dict.word_list
    correct, total, loss_sum, batch_num = 0, 0, 0., 0
    for batch, _, _ in data_reader_eval.batches(one_pass=True):
        batch_res = model.run_batch(sess, batch, train=False, vis=vis)
        if vis and total < cfg.TEST.NUM_VIS:
            vis_batch_vqa(data_reader_eval, batch, batch_res, total, vis_dir)
        if pred:
            predictions.extend([
                {'questionId': q, 'prediction': answer_tokens[p]}
                for q, p in zip(batch['qid_list'], batch_res['predictions'])])
        correct += batch_res['num_correct']
        total += batch_res['batch_size']
        loss_sum += batch_res['loss']
        batch_num += 1
        print('\rEval S %d: avgL=%.4f, avgA=%.4f' % (
            total, loss_sum/batch_num, correct/total), end='')
    print('')
    eval_res = {
        'correct': correct,
        'total': total,
        'accuracy': correct/total,
        'loss': loss_sum/batch_num,
        'predictions': predictions}
    return eval_res


def dump_prediction_to_file(predictions, res_dir):
    pred_file = os.path.join(res_dir, 'pred_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print('predictions written to %s' % pred_file)


def train():
    data_reader_train, num_vocab, num_choices = load_train_data()
    data_reader_eval, _, _ = load_eval_data(max_num=cfg.TRAIN.EVAL_MAX_NUM)

    # Load model
    model = LCGNnet(num_vocab, num_choices, gpusNum=len(cfg.GPUS.split(',')))

    # Save snapshot
    saver_train = tf.train.Saver(max_to_keep=None)  # keep all snapshots
    saver_eval = tf.train.Saver(model.emaDict if cfg.USE_EMA else None)
    snapshot_dir = os.path.dirname(cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, 0))
    os.makedirs(snapshot_dir, exist_ok=True)
    with open(os.path.join(snapshot_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    if cfg.TRAIN.START_EPOCH > 0:
        print('resuming from epoch %d' % cfg.TRAIN.START_EPOCH)
        saver_train.restore(
            sess, cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TRAIN.START_EPOCH))
    else:
        sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    print('%s - train for %d epochs' % (cfg.EXP_NAME, cfg.TRAIN.MAX_EPOCH))
    run_train_on_data(
        model, data_reader_train, saver_train, run_eval=cfg.TRAIN.RUN_EVAL,
        data_reader_eval=data_reader_eval, saver_eval=saver_eval)
    print('%s - train (done)' % cfg.EXP_NAME)


def test():
    data_reader_eval, num_vocab, num_choices = load_eval_data()

    # Load model
    model = LCGNnet(num_vocab, num_choices, gpusNum=len(cfg.GPUS.split(',')))

    # Load test snapshot
    saver_eval = tf.train.Saver(model.emaDict if cfg.USE_EMA else None)
    snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    saver_eval.restore(sess, snapshot_file)
    sess.graph.finalize()

    res_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    vis_dir = os.path.join(
        res_dir, '%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_VQA))
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    pred = cfg.TEST.DUMP_PRED
    if not pred:
        print('NOT writing predictions (set TEST.DUMP_PRED True to write)')

    print('%s - test epoch %d' % (cfg.EXP_NAME, cfg.TEST.EPOCH))
    eval_res = run_eval_on_data(
        model, data_reader_eval, pred=pred, vis=True, vis_dir=vis_dir)
    print('%s - test epoch %d: accuracy = %.4f' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, eval_res['accuracy']))

    # write results
    if pred:
        dump_prediction_to_file(eval_res['predictions'], res_dir)
    eval_res.pop('predictions')
    res_file = os.path.join(res_dir, 'res_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_VQA))
    with open(res_file, 'w') as f:
        json.dump(eval_res, f)


if __name__ == '__main__':
    if cfg.train:
        train()
    else:
        test()
